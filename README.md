# Data_Analysis_Agent

Strategic Analytics Workspace

An AI-powered data audit tool. Upload a CSV or Excel file, and the app cleans it, engineers a few standard business features (profit margin, revenue per unit), and sends a summary to an LLM (via Groq) to generate an 8-stage strategic analysis: dataset understanding, business problem framing, cleaning documentation, EDA, feature engineering notes, insights, a dashboard blueprint, and recommendations — rendered as an interactive report in the browser.

Built with Flask, SQLite, pandas, and Chart.js, with a single-page vanilla-JS frontend.

Features
Email/phone + password authentication (JWT-based sessions)
OTP-based password reset flow
CSV/XLSX upload → automated cleaning (dedup, null handling) → LLM-generated strategic report
Per-user report history with automatic expiry
Single-file frontend, no build step required



Tech stack
Layer	Tech
Backend	- Flask, SQLite, PyJWT, bcrypt
Data -pandas, numpy
AI - Groq API (Llama 3.3 70B)
Frontend -	Vanilla JS, Tailwind (CDN), Chart.js
