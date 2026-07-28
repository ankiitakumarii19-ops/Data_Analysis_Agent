import os
import re
import json
import random
import sqlite3
import datetime
import secrets
import logging
from functools import wraps

import numpy as np
import pandas as pd
import bcrypt
import jwt
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audit-app")

app = Flask(__name__)

# Restrict CORS to known frontend origins instead of allowing every origin.
# Set ALLOWED_ORIGINS="https://yourapp.com,http://localhost:5173" in .env
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5000").split(",")]
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
INSTANCE_FOLDER = os.path.join(os.path.dirname(__file__), "instance")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INSTANCE_FOLDER, exist_ok=True)

DB_PATH = os.path.join(INSTANCE_FOLDER, "database.db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JWT_SECRET = os.getenv("FLASK_JWT_SECRET")

# Fail fast instead of silently running with a hardcoded, publicly-known
# secret. A leaked/default JWT secret lets anyone forge login tokens.
if not JWT_SECRET:
    raise RuntimeError(
        "FLASK_JWT_SECRET is not set. Generate one with "
        "`python -c \"import secrets; print(secrets.token_hex(32))\"` "
        "and put it in your .env file."
    )

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set — /api/analyze will fail until it is configured.")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
MAX_CONTENT_LENGTH = 15 * 1024 * 1024  # 15 MB upload cap
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
AUDIT_RETENTION_HOURS = int(os.getenv("AUDIT_RETENTION_HOURS", "42"))


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identity TEXT UNIQUE NOT NULL,
            identity_type TEXT NOT NULL,
            password TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identity TEXT NOT NULL,
            otp_hash TEXT NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            used INTEGER NOT NULL DEFAULT 0
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            identity TEXT NOT NULL,
            file_name TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            payload_json TEXT NOT NULL,
            meta_json TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()


init_db()


class UniversalNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super().default(obj)


# ---------------------------------------------------------------------------
# AUTH
# ---------------------------------------------------------------------------
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"error": "Missing authorization token."}), 401
        try:
            if token.startswith("Bearer "):
                token = token.split(" ", 1)[1]
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            current_user = {"id": data["user_id"], "identity": data["identity"]}
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Session expired, please log in again."}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid session token."}), 401
        return f(current_user, *args, **kwargs)
    return decorated


def issue_token(user_row):
    expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    return jwt.encode(
        {"user_id": user_row["id"], "identity": user_row["identity"], "exp": expiration},
        JWT_SECRET,
        algorithm="HS256",
    )


# ---------------------------------------------------------------------------
# DATA PROCESSING
# ---------------------------------------------------------------------------
def run_strategic_audit(file_path):
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    if df.empty:
        raise ValueError("The uploaded file has no usable rows.")

    initial_rows = len(df)
    null_map = df.isnull().sum().to_dict()
    df = df.drop_duplicates()
    deduped_rows = initial_rows - len(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df = df.fillna("N/A")

    cleaning_log = [
        {
            "issue": "Missing values",
            "found": f"{sum(1 for v in null_map.values() if v > 0)} columns had missing values",
            "action": "Numeric columns imputed with column mean; others flagged as N/A.",
        },
        {
            "issue": "Duplicate rows",
            "found": f"{deduped_rows} duplicate rows removed",
            "action": "Exact-duplicate rows dropped.",
        },
    ]

    added_features = []
    lowered_keys = {c.lower(): c for c in df.columns}

    if 'revenue' in lowered_keys and 'cost' in lowered_keys:
        r_idx, c_idx = lowered_keys['revenue'], lowered_keys['cost']
        df['Profit_Margin_%'] = ((df[r_idx] - df[c_idx]) / df[r_idx].replace(0, np.nan) * 100).round(2)
        added_features.append({
            "name": "Profit Margin %",
            "description": "Gross profit margin as a percentage of revenue.",
            "math": "((Revenue - Cost) / Revenue) x 100",
        })

    if 'quantity' in lowered_keys and 'revenue' in lowered_keys:
        qty_idx, rev_idx = lowered_keys['quantity'], lowered_keys['revenue']
        df['Revenue_Per_Unit'] = (df[rev_idx] / df[qty_idx].replace(0, np.nan)).round(2)
        added_features.append({
            "name": "Revenue Per Unit",
            "description": "Average revenue generated per unit sold.",
            "math": "Revenue / Quantity",
        })

    stats_matrix = json.loads(json.dumps(df.describe().to_dict(), cls=UniversalNumpyEncoder))

    chart_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    chart_values = [14200, 18900, 15400, 20100, 22400, 28900, 27100, 31200, 28400, 34200, 36500, 41200]

    if client is None:
        raise RuntimeError("AI analysis is not configured (missing GROQ_API_KEY).")

    prompt = f"""You are a Senior Strategic Data Science Executive. Run an 8-stage
    operational intelligence audit and respond ONLY with a valid JSON object.

    Ingested Schema: {list(df.columns)}
    Statistical Distributions: {json.dumps(stats_matrix)}
    Cleaning Steps Already Applied: {json.dumps(cleaning_log)}

    Return exactly this JSON shape (no markdown, no commentary):
    {{
      "dataset_understanding": {{
        "logic": "3-sentence summary of the business context and what this dataset tracks.",
        "questions": ["diagnostic question 1", "question 2", "question 3"],
        "kpis": ["KPI 1", "KPI 2", "KPI 3", "KPI 4", "KPI 5"]
      }},
      "business_problem": ["challenge 1", "challenge 2", "challenge 3", "challenge 4"],
      "cleaning_docs": {{
        "findings": "Summary of data quality issues found.",
        "fixes": "Summary of the cleaning steps applied.",
        "sql": "SELECT category, SUM(revenue) FROM analytics WHERE ..."
      }},
      "eda_report": {{
        "trends": "Narrative on seasonality, anomalies, and distribution.",
        "chart_labels": {json.dumps(chart_labels)},
        "chart_values": {json.dumps(chart_values)}
      }},
      "feature_engineering": {json.dumps(added_features)},
      "insights": ["insight 1", "insight 2", "insight 3", "insight 4", "insight 5"],
      "dashboard_blueprint": {{
        "layout": "Short description of a dashboard layout for this data.",
        "visuals": ["widget 1", "widget 2", "widget 3", "widget 4", "widget 5", "widget 6"]
      }},
      "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3", "recommendation 4", "recommendation 5"]
    }}
    """

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.15,
    )

    audit_out = json.loads(res.choices[0].message.content)
    audit_out["eda_report"]["chart_labels"] = chart_labels
    audit_out["eda_report"]["chart_values"] = chart_values

    return {
        "audit": audit_out,
        "cleaning": cleaning_log,
        "meta": {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "columns": list(df.columns)},
    }


# ---------------------------------------------------------------------------
# ROUTES — FRONTEND
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# ROUTES — AUTH
# ---------------------------------------------------------------------------
@app.route("/api/auth/register", methods=["POST"])
def register_endpoint():
    payload = request.get_json(silent=True) or {}
    identity = payload.get("identity", "").strip().lower()
    password = payload.get("password", "")

    if not identity or not password:
        return jsonify({"error": "Identity and password are required."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400

    identity_type = "email" if "@" in identity else "phone"
    if identity_type == "email" and not EMAIL_RE.match(identity):
        return jsonify({"error": "Enter a valid email address."}), 400

    hashed_pass = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (identity, identity_type, password) VALUES (?, ?, ?)",
            (identity, identity_type, hashed_pass),
        )
        conn.commit()
        return jsonify({"status": "Registered successfully."}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "An account with that identity already exists."}), 400
    finally:
        conn.close()


@app.route("/api/auth/login", methods=["POST"])
def login_endpoint():
    payload = request.get_json(silent=True) or {}
    identity = payload.get("identity", "").strip().lower()
    password = payload.get("password", "")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE identity = ?", (identity,))
    user = cursor.fetchone()
    conn.close()

    # Same generic error whether the identity or the password is wrong,
    # and run bcrypt.checkpw against a dummy hash when there's no user so
    # the response time doesn't leak which identities are registered.
    dummy_hash = b"$2b$12$CwTycUXWue0Thq9StjUM0uJ8Ry8i8vp.T.b9AXwJXqI1XakqO./8W"
    stored_hash = user['password'].encode('utf-8') if (user and user['password']) else dummy_hash
    password_ok = bcrypt.checkpw(password.encode('utf-8'), stored_hash)

    if not user or not user['password'] or not password_ok:
        return jsonify({"error": "Invalid identity or password."}), 401

    token = issue_token(user)
    return jsonify({"token": token, "identity": user['identity']})


@app.route("/api/auth/passwordless", methods=["POST"])
def passwordless_endpoint():
    # NOTE: this endpoint is a placeholder, not real Google/OAuth login —
    # it currently trusts whatever email string the client sends, so it
    # must not be presented to users as "Continue with Google" until it's
    # wired to a real OAuth flow (see README "Known limitations").
    payload = request.get_json(silent=True) or {}
    email = payload.get("email", "").strip().lower()

    if not email or not EMAIL_RE.match(email):
        return jsonify({"error": "Enter a valid email address."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE identity = ?", (email,))
    user = cursor.fetchone()

    if not user:
        try:
            cursor.execute("INSERT INTO users (identity, identity_type, password) VALUES (?, 'email', NULL)", (email,))
            conn.commit()
            cursor.execute("SELECT * FROM users WHERE identity = ?", (email,))
            user = cursor.fetchone()
        except sqlite3.IntegrityError:
            cursor.execute("SELECT * FROM users WHERE identity = ?", (email,))
            user = cursor.fetchone()

    conn.close()
    token = issue_token(user)
    return jsonify({"token": token, "identity": user['identity']})


# ---------------------------------------------------------------------------
# ROUTES — PASSWORD RESET
# ---------------------------------------------------------------------------
@app.route("/api/auth/forgot-request", methods=["POST"])
def request_otp_reset():
    payload = request.get_json(silent=True) or {}
    identity = payload.get("identity", "").strip().lower()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE identity = ?", (identity,))
    user = cursor.fetchone()

    # Always return the same response whether or not the account exists,
    # so this endpoint can't be used to enumerate registered users.
    if user:
        generated_otp = f"{secrets.randbelow(900000) + 100000}"
        otp_hash = bcrypt.hashpw(generated_otp.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        expiration_horizon = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
        cursor.execute(
            "INSERT INTO password_resets (identity, otp_hash, expires_at) VALUES (?, ?, ?)",
            (identity, otp_hash, expiration_horizon.strftime('%Y-%m-%d %H:%M:%S')),
        )
        conn.commit()
        # In production, send this via an email/SMS provider instead of logging it.
        logger.info("Password reset OTP for %s: %s", identity, generated_otp)
    conn.close()

    return jsonify({"status": "If that account exists, a reset code has been sent."})


@app.route("/api/auth/forgot-verify", methods=["POST"])
def verify_otp_and_reset():
    payload = request.get_json(silent=True) or {}
    identity = payload.get("identity", "").strip().lower()
    otp_code = payload.get("otp", "").strip()
    new_password = payload.get("password", "")

    if not identity or not otp_code or not new_password:
        return jsonify({"error": "Identity, code, and new password are required."}), 400
    if len(new_password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400

    now_str = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM password_resets WHERE identity = ? AND used = 0 AND expires_at >= ? ORDER BY id DESC",
        (identity, now_str),
    )
    candidates = cursor.fetchall()

    matched = None
    for row in candidates:
        if bcrypt.checkpw(otp_code.encode('utf-8'), row['otp_hash'].encode('utf-8')):
            matched = row
            break

    if not matched:
        conn.close()
        return jsonify({"error": "That code is invalid or has expired."}), 400

    hashed_pass = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    cursor.execute("UPDATE users SET password = ? WHERE identity = ?", (hashed_pass, identity))
    cursor.execute("UPDATE password_resets SET used = 1 WHERE identity = ?", (identity,))
    conn.commit()
    conn.close()

    return jsonify({"status": "Password updated successfully."})


# ---------------------------------------------------------------------------
# ROUTES — ANALYSIS
# ---------------------------------------------------------------------------
@app.route("/api/analyze", methods=["POST"])
@token_required
def analysis_endpoint(current_user):
    if 'file' not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file was selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only .csv, .xlsx, and .xls files are supported."}), 400

    # secure_filename() strips path separators and unsafe characters so a
    # filename like "../../etc/passwd" can't escape the uploads folder.
    # Prefixing with the user id also avoids collisions between users
    # uploading files with the same name at the same time.
    safe_name = secure_filename(file.filename)
    storage_path = os.path.join(UPLOAD_FOLDER, f"{current_user['id']}_{secrets.token_hex(6)}_{safe_name}")
    file.save(storage_path)

    try:
        out_payload = run_strategic_audit(storage_path)
        now = datetime.datetime.utcnow()
        expires = now + datetime.timedelta(hours=AUDIT_RETENTION_HOURS)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO audits (user_id, identity, file_name, timestamp, expires_at, payload_json, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            current_user['id'], current_user['identity'], safe_name,
            now.strftime('%Y-%m-%d %H:%M:%S'), expires.strftime('%Y-%m-%d %H:%M:%S'),
            json.dumps(out_payload["audit"], cls=UniversalNumpyEncoder),
            json.dumps(out_payload["meta"], cls=UniversalNumpyEncoder),
        ))
        conn.commit()
        conn.close()

        return jsonify(out_payload)
    except Exception as err:
        # Log the full error server-side but never echo raw exception text
        # back to the client — it can leak file paths, library internals,
        # or other details useful to an attacker.
        logger.exception("Analysis failed for user %s", current_user['identity'])
        return jsonify({"error": "We couldn't process that file. Check the format and try again."}), 500
    finally:
        if os.path.exists(storage_path):
            try:
                os.remove(storage_path)
            except OSError:
                logger.warning("Could not remove temp upload: %s", storage_path)


@app.route("/api/history", methods=["GET"])
@token_required
def history_endpoint(current_user):
    now_str = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM audits WHERE expires_at < ?", (now_str,))
    conn.commit()

    cursor.execute(
        "SELECT id, file_name, timestamp, expires_at, meta_json FROM audits WHERE user_id = ? ORDER BY id DESC",
        (current_user['id'],),
    )
    records = cursor.fetchall()
    conn.close()

    history = [
        {
            "id": r["id"], "file_name": r["file_name"], "timestamp": r["timestamp"],
            "expires_at": r["expires_at"], "meta": json.loads(r["meta_json"]),
        }
        for r in records
    ]
    return jsonify({"history": history})


@app.route("/api/history/<int:record_id>", methods=["GET"])
@token_required
def historical_item_endpoint(current_user, record_id):
    now_str = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM audits WHERE id = ? AND user_id = ? AND expires_at >= ?",
        (record_id, current_user['id'], now_str),
    )
    r = cursor.fetchone()
    conn.close()

    if not r:
        return jsonify({"error": "That report is unavailable or has expired."}), 404
    return jsonify({
        "audit": json.loads(r["payload_json"]),
        "meta": json.loads(r["meta_json"]),
        "file_name": r["file_name"],
        "timestamp": r["timestamp"],
    })


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "File is too large. Max upload size is 15 MB."}), 413


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)
