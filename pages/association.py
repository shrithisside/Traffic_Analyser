                        #Name: Shriwayanta Maiti, PRN: 20240802169
                        #Name: Harshal Nanekar, PRN: 20240802163


"""
Association Rules Mining page — Apriori algorithm.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    from modules.association import run_association_rules, MLXTEND_AVAILABLE
except ImportError:
    MLXTEND_AVAILABLE = False


def render(df: pd.DataFrame):
    st.markdown(
        """
        <div class="page-header">
            <h1>🔗 Association Rules Mining</h1>
            <p>Discover hidden co-occurrence patterns between violations using the Apriori algorithm</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Library check ─────────────────────────────────────────────────────────
    if not MLXTEND_AVAILABLE:
        st.error("⚠️ `mlxtend` library not installed.")
        st.code("pip install mlxtend", language="bash")
        st.info("Add `mlxtend` to your `requirements.txt` and restart the app.")
        return

    if df.empty or len(df) < 20:
        st.warning("Need at least 20 records to mine association rules.")
        return

    # ── Theory expander ───────────────────────────────────────────────────────
    with st.expander("📚 How Apriori Algorithm Works", expanded=False):
        st.markdown("""
        **Apriori** finds frequent itemsets then derives association rules from them.

        Each **transaction** = all violation types that occurred on the same day in the same area.

        | Metric | Formula | Meaning |
        |---|---|---|
        | **Support** | freq(A∪B) / N | How often A and B appear together |
        | **Confidence** | freq(A∪B) / freq(A) | Given A, how likely is B |
        | **Lift** | Confidence / Support(B) | Strength above random chance |

        - **Lift > 1** → A and B are positively correlated (appear together more than random)
        - **Lift = 1** → Independent
        - **Lift < 1** → Negatively correlated
        """)

    # ── Controls ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⚙️ Parameters</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_support = st.slider(
            "Min Support", 0.01, 0.5, 0.05, 0.01,
            help="Minimum fraction of transactions containing the itemset",
            key="ar_sup",
        )
    with c2:
        min_confidence = st.slider(
            "Min Confidence", 0.1, 1.0, 0.4, 0.05,
            help="Minimum probability that B occurs given A",
            key="ar_conf",
        )
    with c3:
        min_lift = st.slider(
            "Min Lift", 1.0, 5.0, 1.0, 0.1,
            help="Minimum strength above random chance",
            key="ar_lift",
        )
    with c4:
        mine_by = st.selectbox(
            "Mine By",
            ["violation_type", "area"],
            help="Find patterns among violation types or among areas",
            key="ar_by",
        )

    run_btn = st.button("🚀 Run Apriori Mining", use_container_width=True)

    if "assoc_results" not in st.session_state or run_btn:
        with st.spinner("⛏️ Mining association rules..."):
            freq_items, rules = run_association_rules(
                df,
                min_support=min_support,
                min_confidence=min_confidence,
                min_lift=min_lift,
                by=mine_by,
            )
            st.session_state["assoc_results"] = (freq_items, rules)

    freq_items, rules = st.session_state["assoc_results"]

    if freq_items.empty:
        st.warning("No frequent itemsets found. Try **lowering Min Support** (e.g. 0.02).")
        return

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f"""<div class="metric-card" style="--card-accent:linear-gradient(90deg,#a78bfa,#7c3aed)">
                <span class="metric-icon">📦</span>
                <div class="metric-value">{len(freq_items)}</div>
                <div class="metric-label">Frequent Itemsets</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""<div class="metric-card" style="--card-accent:linear-gradient(90deg,#60a5fa,#2563eb)">
                <span class="metric-icon">📏</span>
                <div class="metric-value">{len(rules)}</div>
                <div class="metric-label">Rules Generated</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with k3:
        avg_lift = rules["lift"].mean() if not rules.empty else 0.0
        st.markdown(
            f"""<div class="metric-card" style="--card-accent:linear-gradient(90deg,#34d399,#059669)">
                <span class="metric-icon">⬆️</span>
                <div class="metric-value">{avg_lift:.2f}</div>
                <div class="metric-label">Average Lift</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Frequent itemsets bar chart ───────────────────────────────────────────
    st.markdown('<div class="section-title">📦 Top Frequent Itemsets</div>', unsafe_allow_html=True)
    freq_sorted = freq_items.sort_values("support", ascending=False).head(20)
    fig_freq = px.bar(
        freq_sorted,
        x="support",
        y="itemsets",
        orientation="h",
        color="support",
        color_continuous_scale="Plasma",
        title="Top 20 Frequent Itemsets by Support",
        labels={"support": "Support", "itemsets": "Itemset"},
    )
    fig_freq.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        coloraxis_showscale=False,
        height=max(400, len(freq_sorted) * 28),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_freq, use_container_width=True)

    # ── Rules section ─────────────────────────────────────────────────────────
    if rules.empty:
        st.info("No rules generated at current thresholds. Try lowering Confidence or Lift.")
        return

    st.markdown('<div class="section-title">📋 Association Rules Table</div>', unsafe_allow_html=True)
    rules_display = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    rules_display.columns = ["If  (Antecedent)", "Then  (Consequent)", "Support", "Confidence", "Lift"]
    rules_display = rules_display.sort_values("Lift", ascending=False).reset_index(drop=True)
    st.dataframe(rules_display, use_container_width=True, hide_index=True)

    # ── Scatter: Confidence vs Lift ───────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Confidence vs Lift</div>', unsafe_allow_html=True)
    fig_scatter = px.scatter(
        rules_display,
        x="Confidence",
        y="Lift",
        size="Support",
        color="Lift",
        hover_data=["If  (Antecedent)", "Then  (Consequent)"],
        color_continuous_scale="Viridis",
        title="Rules Quality: Confidence vs Lift  (bubble size = Support)",
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Top rule highlights ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">💡 Top Discovered Rules</div>', unsafe_allow_html=True)
    for _, row in rules_display.head(5).iterrows():
        conf_pct = f"{row['Confidence']:.1%}"
        st.markdown(
            f"""<div class="insight-card">
                <div class="ic-label">
                    Lift: <b style="color:#a78bfa">{row['Lift']:.2f}</b> &nbsp;|&nbsp;
                    Confidence: <b style="color:#60a5fa">{conf_pct}</b> &nbsp;|&nbsp;
                    Support: <b style="color:#34d399">{row['Support']:.3f}</b>
                </div>
                <div class="ic-value" style="font-size:0.95rem;margin-top:6px;">
                    <span style="color:#f59e0b">IF </span>
                    <span style="color:#e2e8f0">{row['If  (Antecedent)']}</span>
                    <span style="color:#64748b"> &nbsp;→&nbsp; </span>
                    <span style="color:#f59e0b">THEN </span>
                    <span style="color:#34d399">{row['Then  (Consequent)']}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )