                        #Name: Shriwayanta Maiti, PRN: 20240802169
                        #Name: Harshal Nanekar, PRN: 20240802163


"""
CSS injection for dark premium theme.
"""
import streamlit as st


def inject_css():
    st.markdown(
        """
<style>
[data-testid="stSidebarNav"] {
    display: none;
}
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── reset & base ─────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Dark background */
.stApp {
    background: #0a0f1e;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1226 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}

/* ── sidebar logo ─────────────────────────────────── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 4px 16px;
}
.logo-icon { font-size: 2.2rem; }
.logo-title {
    font-size: 1.25rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.logo-sub {
    font-size: 0.7rem;
    font-weight: 500;
    color: #64748b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── sidebar stats ────────────────────────────────── */
.sidebar-stats {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.stat-item {
    background: #1e293b;
    border-radius: 10px;
    padding: 10px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid #334155;
}
.stat-num {
    font-size: 1.3rem;
    font-weight: 700;
    color: #a78bfa;
}
.stat-label {
    font-size: 0.75rem;
    color: #94a3b8;
}

/* ── page header ──────────────────────────────────── */
.page-header {
    margin-bottom: 28px;
}
.page-header h1 {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 4px;
}
.page-header p {
    color: #64748b;
    font-size: 0.9rem;
    margin: 0;
}

/* ── metric cards ─────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(167,139,250,0.15);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--card-accent, linear-gradient(90deg,#a78bfa,#60a5fa));
    border-radius: 16px 16px 0 0;
}
.metric-icon {
    font-size: 1.8rem;
    margin-bottom: 8px;
    display: block;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── insight card ─────────────────────────────────── */
.insight-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.insight-card .ic-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.insight-card .ic-value {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-top: 2px;
}

/* ── risk badge ───────────────────────────────────── */
.risk-high   { color: #ef4444; font-weight: 700; }
.risk-medium { color: #f59e0b; font-weight: 700; }
.risk-low    { color: #22c55e; font-weight: 700; }

/* ── section titles ───────────────────────────────── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #cbd5e1;
    border-left: 3px solid #a78bfa;
    padding-left: 10px;
    margin: 24px 0 14px;
}

/* ── warehouse table ──────────────────────────────── */
.wh-table { border-radius: 12px; overflow: hidden; }

/* ── form card ────────────────────────────────────── */
.form-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 20px;
    padding: 32px;
}

/* Streamlit specific overrides */
[data-testid="stRadio"] label {
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
    padding: 6px 0 !important;
    cursor: pointer;
}
[data-testid="stRadio"] label:hover { color: #e2e8f0 !important; }

[data-testid="stButton"] button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: white !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(124,58,237,0.4) !important;
}

/* Input fields */
[data-testid="stSelectbox"] > div,
[data-testid="stDateInput"] > div input,
[data-testid="stSlider"] {
    background: #1e293b !important;
    border-color: #334155 !important;
    color: #e2e8f0 !important;
}

/* Divider */
hr { border-color: #1e293b !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    background: #111827 !important;
    border-radius: 12px !important;
}

/* Success/error */
[data-testid="stSuccess"] { background: rgba(34,197,94,0.1) !important; border-radius: 8px; }
[data-testid="stError"]   { background: rgba(239,68,68,0.1) !important; border-radius: 8px; }
</style>
""",
        unsafe_allow_html=True,
    )