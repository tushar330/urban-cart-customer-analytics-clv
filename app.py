#!/usr/bin/env python3
"""UrbanCart analytics — five Olist-backed dashboards with formula hover."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ui.components import (  # noqa: E402
    CLV_COLORS,
    INK,
    MUTED,
    RFM_COLORS,
    RISK_COLORS,
    TEAL,
    brl,
    chart_card,
    footer,
    inject_theme,
    kpi_band,
    lede,
    masthead,
    polish_hbar,
    polish_vbar,
    table_card,
)

DB_URL = os.getenv(
    "URBANCART_DB_URL",
    f"postgresql+psycopg2://{os.getenv('USER', 'postgres')}@localhost:5432/urbancart",
)

RFM_ORDER = [
    "Champions",
    "Loyal Customers",
    "New Customers",
    "Average Customers",
    "At Risk",
    "Lost Customers",
]
RISK_ORDER = ["High Risk", "Medium Risk", "Low Risk"]
CLV_ORDER = ["High Value", "Medium Value", "Low Value"]

# Exact CASE rules from setup_local.py → rfm_segmentation
RFM_SEGMENT_RULES = {
    "Champions": {
        "how": (
            "Most recent, most frequent, highest spend. Applied first in the CASE ladder. "
            "Scores: Recency and Monetary are NTILE(5) quintiles; Frequency is rule-based "
            "(1 order → 1, 2 orders → 3, 3+ orders → 5) because most Olist buyers order once."
        ),
        "formula": (
            "Champions when:\n"
            "  Recency score ≥ 4\n"
            "  AND Frequency score = 5\n"
            "  AND Monetary score ≥ 4"
        ),
    },
    "Loyal Customers": {
        "how": (
            "Recent-enough buyers who ordered more than once. Checked after Champions. "
            "Does not require a high Monetary score."
        ),
        "formula": (
            "Loyal Customers when (else if):\n"
            "  Recency score ≥ 3\n"
            "  AND Frequency score ≥ 3"
        ),
    },
    "New Customers": {
        "how": (
            "Recent first-time buyers (Frequency score 1 = exactly one order). "
            "Checked after Champions and Loyal."
        ),
        "formula": (
            "New Customers when (else if):\n"
            "  Recency score ≥ 4\n"
            "  AND Frequency score = 1"
        ),
    },
    "At Risk": {
        "how": (
            "Previously active (2+ orders) but not recent. Checked after New Customers."
        ),
        "formula": (
            "At Risk when (else if):\n"
            "  Recency score ≤ 2\n"
            "  AND Frequency score ≥ 3"
        ),
    },
    "Lost Customers": {
        "how": (
            "One-time buyers who have not returned and spent little. "
            "Checked after At Risk."
        ),
        "formula": (
            "Lost Customers when (else if):\n"
            "  Recency score ≤ 2\n"
            "  AND Frequency score = 1\n"
            "  AND Monetary score ≤ 2"
        ),
    },
    "Average Customers": {
        "how": (
            "Everyone who did not match Champions, Loyal, New, At Risk, or Lost "
            "(the ELSE branch of the segment CASE)."
        ),
        "formula": (
            "Average Customers when:\n"
            "  ELSE (no earlier RFM rule matched)"
        ),
    },
}

CHURN_SEGMENT_RULES = {
    "High Risk": {
        "how": "Buyers inactive longer than 180 days relative to the dataset’s last order date (2018-09-03).",
        "formula": "High Risk when:\n  days_inactive > 180\nwhere days_inactive = analysis_date − last_order",
    },
    "Medium Risk": {
        "how": "Buyers inactive more than 90 days but not more than 180 days.",
        "formula": "Medium Risk when:\n  days_inactive > 90\n  AND days_inactive ≤ 180",
    },
    "Low Risk": {
        "how": "Buyers who ordered within the last 90 days of the analysis date.",
        "formula": "Low Risk when:\n  days_inactive ≤ 90",
    },
}

CLV_SEGMENT_RULES = {
    "Low Value": {
        "how": "Historic customer lifetime value under R$ 150 (BRL bands for Olist order scale).",
        "formula": "Low Value when:\n  Customer lifetime value < 150\nCLV = Average order value × purchase frequency × lifespan_years",
    },
    "Medium Value": {
        "how": "Historic customer lifetime value from R$ 150 up to (but not including) R$ 500.",
        "formula": "Medium Value when:\n  150 ≤ Customer lifetime value < 500",
    },
    "High Value": {
        "how": "Historic customer lifetime value of R$ 500 or more.",
        "formula": "High Value when:\n  Customer lifetime value ≥ 500",
    },
}

RFM_SCORE_RULES = (
    "Score build (shared by all segments):\n"
    "  Recency score = NTILE(5) ORDER BY days_inactive DESC  (5 = most recent)\n"
    "  Frequency score = 1 if orders=1; 3 if orders=2; 5 if orders≥3\n"
    "  Monetary score = NTILE(5) ORDER BY Σ amount ASC  (5 = highest spend)\n"
    "Segment CASE order: Champions → Loyal → New → At Risk → Lost → ELSE Average"
)

# Brazilian UF codes → full state names (Olist ships abbreviations only).
BR_STATE_NAMES = {
    "AC": "Acre",
    "AL": "Alagoas",
    "AP": "Amapá",
    "AM": "Amazonas",
    "BA": "Bahia",
    "CE": "Ceará",
    "DF": "Distrito Federal",
    "ES": "Espírito Santo",
    "GO": "Goiás",
    "MA": "Maranhão",
    "MT": "Mato Grosso",
    "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais",
    "PA": "Pará",
    "PB": "Paraíba",
    "PR": "Paraná",
    "PE": "Pernambuco",
    "PI": "Piauí",
    "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul",
    "RO": "Rondônia",
    "RR": "Roraima",
    "SC": "Santa Catarina",
    "SP": "São Paulo",
    "SE": "Sergipe",
    "TO": "Tocantins",
}


@st.cache_resource
def get_engine():
    return create_engine(DB_URL)


@st.cache_data(ttl=120)
def read_sql(query: str) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn)


def fig_layout(fig: go.Figure, *, legend: bool = True) -> go.Figure:
    # Do not overwrite hovertemplates — callers set their own.
    fig.update_layout(showlegend=legend, hovermode="closest")
    return fig


def product_display_names(df: pd.DataFrame) -> pd.DataFrame:
    """Olist publishes no product titles — only category. Name = GMV rank + English category."""
    ranked = df.sort_values("revenue", ascending=False).reset_index(drop=True)
    ranked["product_id"] = ranked["product_id"].astype(str)
    ranked["_rank"] = range(1, len(ranked) + 1)
    within = ranked.groupby("category", sort=False).cumcount() + 1
    totals = ranked.groupby("category")["product_id"].transform("count")
    labels = []
    for i, row in ranked.iterrows():
        cat = str(row["category"]).replace("_", " ")
        label = f"{int(row['_rank'])}. {cat}"
        if int(totals.loc[i]) > 1:
            label = f"{label} #{int(within.loc[i])}"
        labels.append(label)
    ranked["product_name"] = labels
    return ranked


# ---------------------------------------------------------------------------
# 1. Executive
# ---------------------------------------------------------------------------

def page_executive() -> None:
    masthead(
        "Ledger",
        "01  /  Executive overview",
        [
            "UrbanCart  ·  Olist public orders",
            "Brazil  ·  Sep 2016 – Sep 2018",
        ],
    )
    lede(
        "A snapshot of the book: what the marketplace earned, who bought, "
        "and how much of the base is already silent."
    )

    k = read_sql(
        """
        SELECT
            (SELECT SUM(amount) FROM orders_fact) AS revenue,
            (SELECT COUNT(DISTINCT order_id) FROM orders_fact) AS orders,
            (SELECT COUNT(*) FROM dim_customers) AS customers,
            (SELECT AVG(clv) FROM clv_table) AS avg_clv,
            (SELECT COUNT(*) FILTER (WHERE churn_segment = 'High Risk')::numeric
                 / NULLIF(COUNT(*), 0)
             FROM churn_risk) AS churn_rate
        """
    ).iloc[0]
    aov = float(k.revenue) / float(k.orders)

    kpi_band(
        [
            {
                "label": "Gross merchandise value",
                "value": brl(float(k.revenue)),
                "sub": "price + freight, kept orders",
                "how": "Sum of line amount on every kept order line. Amount is item price plus freight.",
                "formula": "Gross merchandise value = Σ (price + freight_value)",
                "source": "orders_fact.amount",
            },
            {
                "label": "Orders",
                "value": f"{int(k.orders):,}",
                "sub": "distinct order_id",
                "how": "Count of unique orders after dropping canceled and unavailable statuses.",
                "formula": "Orders = COUNT(DISTINCT order_id)",
                "source": "orders_fact",
            },
            {
                "label": "Buyers",
                "value": f"{int(k.customers):,}",
                "sub": "unique people (customer_unique_id)",
                "how": "Olist issues a new customer_id per order. We use customer_unique_id so one person is one row.",
                "formula": "Buyers = COUNT(DISTINCT customer_id)",
                "source": "dim_customers",
            },
            {
                "label": "Average order value",
                "value": brl(aov),
                "sub": "revenue / orders",
                "how": "Average order value across the whole book.",
                "formula": "Average order value = Gross merchandise value / Orders",
                "source": "orders_fact",
            },
            {
                "label": "Average customer lifetime value",
                "value": brl(float(k.avg_clv)),
                "sub": "historic, per buyer",
                "how": "Mean of customer-level lifetime value. One-order buyers get a 1-day floor so lifetime value equals that order.",
                "formula": "Customer lifetime value = Average order value × frequency × lifespan_years",
                "source": "clv_table",
            },
            {
                "label": "Churn rate",
                "value": f"{float(k.churn_rate)*100:.1f}%",
                "sub": "share High Risk (>180 days)",
                "how": (
                    "Share of buyers in High Risk: inactive more than 180 days before the dataset’s "
                    "last order date (not today’s calendar)."
                ),
                "formula": (
                    "Churn rate = COUNT(High Risk) / COUNT(Buyers)\n"
                    "High Risk: days_inactive > 180\n"
                    "days_inactive = analysis_date − last_order"
                ),
                "source": "churn_risk",
            },
        ]
    )

    monthly = read_sql("SELECT * FROM monthly_kpis ORDER BY month")
    monthly["month"] = pd.to_datetime(monthly["month"])

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly["revenue"],
                fill="tozeroy",
                line=dict(color=TEAL, width=2),
                fillcolor="rgba(31,111,120,0.12)",
                name="Revenue",
                hovertemplate="%{x|%b %Y}<br>R$ %{y:,.0f}<extra></extra>",
            )
        )
        fig_layout(fig, legend=False)
        fig.update_yaxes(tickprefix="R$ ", separatethousands=True)
        chart_card(
            "Monthly revenue",
            fig,
            how="Gross merchandise value by purchase month. Canceled and unavailable orders are excluded.",
            formula="Revenue_m = Σ amount  GROUP BY month(order_date)",
            source="monthly_kpis ← orders_fact",
            kind="Area chart",
        )
    with right:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=monthly["month"],
                y=monthly["orders"],
                name="Orders",
                marker_color=TEAL,
                opacity=0.85,
                hovertemplate="%{x|%b %Y}<br>%{y:,} orders<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly["customers"],
                name="Buyers",
                yaxis="y2",
                line=dict(color="#B8573E", width=2),
                hovertemplate="%{x|%b %Y}<br>%{y:,} buyers<extra></extra>",
            )
        )
        fig.update_layout(
            yaxis=dict(title="Orders", automargin=True),
            yaxis2=dict(title="Buyers", overlaying="y", side="right", showgrid=False, automargin=True),
            showlegend=True,
            legend=dict(orientation="h", y=-0.2, x=0),
            margin=dict(l=56, r=56, t=16, b=64),
        )
        chart_card(
            "Orders vs buyers",
            fig,
            how="Order volume against distinct people purchasing that month. A gap means the same buyers placed multiple orders.",
            formula="Orders_m = COUNT DISTINCT order_id\nBuyers_m = COUNT DISTINCT customer_id",
            source="monthly_kpis",
            kind="Combo chart (bar + line)",
        )

    st.write("")
    left, right = st.columns(2)
    with left:
        states = read_sql(
            "SELECT state, revenue, orders, customers FROM state_revenue ORDER BY revenue DESC LIMIT 10"
        )
        states = states.copy()
        states["state_name"] = states["state"].map(BR_STATE_NAMES).fillna(states["state"])
        fig = px.bar(
            states.sort_values("revenue"),
            x="revenue",
            y="state_name",
            orientation="h",
        )
        fig.update_traces(
            marker_color=TEAL,
            hovertemplate="<b>%{y}</b><br>R$ %{x:,.0f}<extra></extra>",
        )
        fig.update_xaxes(tickprefix="R$ ")
        fig_layout(fig, legend=False)
        polish_hbar(fig, left=140)
        chart_card(
            "Revenue by state — top 10",
            fig,
            how="Where merchandise value concentrates. State is the buyer’s full shipping state name (mapped from Olist UF codes) on their first kept order.",
            formula="Revenue_state = Σ amount  GROUP BY customer_state",
            source="state_revenue ← orders_fact ⋈ dim_customers",
            kind="Horizontal bar chart",
        )
    with right:
        clv = read_sql(
            """
            SELECT customer_segment, COUNT(*) AS customers, ROUND(AVG(clv)::numeric, 2) AS avg_clv
            FROM clv_segmentation GROUP BY 1
            """
        )
        clv["customer_segment"] = pd.Categorical(clv["customer_segment"], CLV_ORDER, ordered=True)
        clv = clv.sort_values("customer_segment")
        fig = go.Figure(
            go.Pie(
                labels=clv["customer_segment"],
                values=clv["customers"],
                hole=0.62,
                marker=dict(colors=[CLV_COLORS[s] for s in clv["customer_segment"]]),
                textinfo="percent",
                hovertemplate="%{label}<br>%{value:,} buyers<extra></extra>",
            )
        )
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", y=-0.12, x=0),
            margin=dict(l=16, r=16, t=16, b=56),
        )
        chart_card(
            "Customer lifetime value mix",
            fig,
            how=(
                "Share of buyers in each historic lifetime-value band (BRL, tuned to Olist). "
                "Lifetime value = Average order value × purchase frequency × lifespan_years "
                "(lifespan floored at 1 day so one-order buyers are not zero)."
            ),
            formula=(
                "Low Value: Customer lifetime value < 150\n"
                "Medium Value: 150 ≤ Customer lifetime value < 500\n"
                "High Value: Customer lifetime value ≥ 500\n"
                "Customer lifetime value = Average order value × (orders / lifespan_years) × lifespan_years"
            ),
            source="clv_segmentation ← clv_table",
            kind="Donut chart",
        )


# ---------------------------------------------------------------------------
# 2. RFM
# ---------------------------------------------------------------------------

def page_rfm() -> None:
    masthead(
        "Who still buys",
        "02  /  RFM segmentation",
        ["Recency · Frequency · Monetary", "Scores 1–5, analysis date = last order"],
    )
    lede(
        "Quintiles on recency and spend. Frequency is rule-based: almost every "
        "Olist buyer purchases once, so quintiles on frequency would collapse."
    )

    summary = read_sql(
        """
        SELECT customer_segment,
               COUNT(*) AS customers,
               ROUND(AVG(recency)::numeric, 1) AS avg_recency,
               ROUND(AVG(frequency)::numeric, 2) AS avg_frequency,
               SUM(monetary) AS total_monetary
        FROM rfm_segmentation
        GROUP BY 1
        """
    )
    summary["customer_segment"] = pd.Categorical(summary["customer_segment"], RFM_ORDER, ordered=True)
    summary = summary.sort_values("customer_segment")

    kpi_band(
        [
            {
                "label": row.customer_segment,
                "value": f"{int(row.customers):,}",
                "sub": f"{brl(float(row.total_monetary))} merchandise value",
                "how": RFM_SEGMENT_RULES[str(row.customer_segment)]["how"],
                "formula": (
                    RFM_SEGMENT_RULES[str(row.customer_segment)]["formula"]
                    + "\n\n"
                    + RFM_SCORE_RULES
                ),
                "source": "rfm_segmentation",
            }
            for _, row in summary.iterrows()
        ]
    )

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            summary,
            x="customers",
            y="customer_segment",
            orientation="h",
            color="customer_segment",
            color_discrete_map=RFM_COLORS,
            category_orders={"customer_segment": RFM_ORDER},
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} buyers<extra></extra>")
        fig.update_layout(showlegend=False)
        polish_hbar(fig, left=130)
        chart_card(
            "Segment headcount",
            fig,
            how="Buyer count per Recency–Frequency–Monetary segment. Rules are applied in CASE order (Champions first, Average last).",
            formula=(
                "Champions: Recency≥4 AND Frequency=5 AND Monetary≥4\n"
                "Loyal Customers: Recency≥3 AND Frequency≥3\n"
                "New Customers: Recency≥4 AND Frequency=1\n"
                "At Risk: Recency≤2 AND Frequency≥3\n"
                "Lost Customers: Recency≤2 AND Frequency=1 AND Monetary≤2\n"
                "Average Customers: ELSE"
            ),
            source="rfm_segmentation",
            kind="Horizontal bar chart",
        )
    with right:
        fig = px.bar(
            summary,
            x="customer_segment",
            y="total_monetary",
            color="customer_segment",
            color_discrete_map=RFM_COLORS,
            category_orders={"customer_segment": RFM_ORDER},
        )
        fig.update_traces(hovertemplate="<b>%{x}</b><br>R$ %{y:,.0f}<extra></extra>")
        fig.update_layout(showlegend=False)
        fig.update_yaxes(tickprefix="R$ ")
        polish_vbar(fig, tickangle=-20)
        chart_card(
            "Revenue by segment",
            fig,
            how="Sum of each buyer’s historic spend, rolled up by the same RFM segment rules as the KPI band.",
            formula=(
                "Revenue_segment = Σ monetary  GROUP BY customer_segment\n"
                "monetary = Σ amount for that buyer\n"
                "Same CASE rules: Champions / Loyal / New / At Risk / Lost / Average"
            ),
            source="rfm_segmentation.monetary",
            kind="Vertical bar chart",
        )

    heat = read_sql(
        """
        SELECT r_score, f_score, COUNT(*) AS customers
        FROM rfm_segmentation
        GROUP BY 1, 2
        """
    )
    pivot = heat.pivot(index="f_score", columns="r_score", values="customers").fillna(0)
    pivot = pivot.reindex(index=[1, 3, 5], columns=[1, 2, 3, 4, 5], fill_value=0)
    counts = pivot.values.astype(float)
    # Linear colour washes out Frequency 3/5 (~30–600) next to Frequency 1 (~18k).
    # Colour on log1p(count); label and hover keep the real headcount.
    z_color = np.log1p(counts)
    text = [[f"{int(v):,}" for v in row] for row in counts]
    tick_counts = [50, 500, 5_000, 18_000]
    fig = go.Figure(
        go.Heatmap(
            z=z_color,
            x=[f"Recency {c}" for c in pivot.columns],
            y=[f"Frequency {i}" for i in pivot.index],
            customdata=counts,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#1F2A2E"),
            colorscale=[
                [0.00, "#F4EFE6"],
                [0.25, "#C5D9D4"],
                [0.50, "#6FA8A8"],
                [0.75, "#2F7F86"],
                [1.00, "#154F56"],
            ],
            colorbar=dict(
                title=dict(text="Buyers", side="right"),
                tickmode="array",
                tickvals=[np.log1p(v) for v in tick_counts],
                ticktext=[f"{v:,}" for v in tick_counts],
                thickness=14,
                len=0.85,
            ),
            hovertemplate="%{x} × %{y}<br>%{customdata:,.0f} buyers<extra></extra>",
        )
    )
    fig.update_layout(margin=dict(l=88, r=72, t=16, b=48))
    fig.update_xaxes(title_text="Recency score", automargin=True, side="bottom")
    fig.update_yaxes(title_text="Frequency score", automargin=True, autorange="reversed")
    chart_card(
        "Recency × Frequency heatmap",
        fig,
        how=(
            "Cell = number of buyers with that Recency score and Frequency score. "
            "Colour uses a log scale so rare Frequency 3/5 cells stay visible next to "
            "huge Frequency 1 bands; the number in each cell is the real count. "
            "Segments also use Monetary (see segment i tips)."
        ),
        formula=(
            "Recency score = NTILE(5) ORDER BY (analysis_date − last_order) DESC\n"
            "Frequency score = 1 if orders=1; 3 if orders=2; 5 if orders≥3\n"
            "Cell value = COUNT(*) GROUP BY Recency score, Frequency score\n"
            "Colour = log(1 + cell value)"
        ),
        source="rfm_segmentation",
        kind="Heatmap",
        height=360,
    )

    money = read_sql("SELECT customer_segment, monetary FROM rfm_segmentation")
    money["customer_segment"] = pd.Categorical(money["customer_segment"], RFM_ORDER, ordered=True)
    fig = px.box(
        money,
        x="customer_segment",
        y="monetary",
        color="customer_segment",
        color_discrete_map=RFM_COLORS,
        category_orders={"customer_segment": RFM_ORDER},
        points=False,
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(tickprefix="R$ ", type="log")
    polish_vbar(fig, tickangle=-20)
    chart_card(
        "Monetary spread by segment",
        fig,
        how=(
            "Distribution of lifetime spend inside each RFM segment (same CASE rules as the KPIs). "
            "Log scale because a few Champions dwarf the rest."
        ),
        formula=(
            "monetary = Σ amount per buyer\n"
            "Box plot of monetary GROUP BY customer_segment\n"
            "Segments: Champions / Loyal / New / At Risk / Lost / Average (ELSE)"
        ),
        source="rfm_segmentation.monetary",
        kind="Box plot",
        height=360,
    )

    table_card(
        "Segment table",
        how=(
            "One row per RFM segment with headcount, typical recency, typical frequency, and total spend. "
            "Segment membership uses the same CASE rules as the KPI i tips."
        ),
        formula=(
            "avg_recency = AVG(analysis_date − last_order)\n"
            "avg_frequency = AVG(order count)\n"
            "Champions: Recency≥4, Frequency=5, Monetary≥4\n"
            "Loyal: Recency≥3, Frequency≥3 · New: Recency≥4, Frequency=1\n"
            "At Risk: Recency≤2, Frequency≥3 · Lost: Recency≤2, Frequency=1, Monetary≤2\n"
            "Average: ELSE"
        ),
        source="rfm_segmentation",
        kind="Data table",
    )
    show = summary.rename(
        columns={
            "customer_segment": "Segment",
            "customers": "Buyers",
            "avg_recency": "Avg recency (days)",
            "avg_frequency": "Avg frequency",
            "total_monetary": "Merchandise value (BRL)",
        }
    )
    st.dataframe(show, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# 3. Churn
# ---------------------------------------------------------------------------

def page_churn() -> None:
    masthead(
        "Who went quiet",
        "03  /  Churn risk",
        ["Clock = last order in the extract", "High > 180 days  ·  Medium > 90"],
    )
    lede(
        "Inactivity from the dataset’s last purchase date — not today’s calendar, "
        "or every 2018 buyer would look lost."
    )

    mix = read_sql(
        """
        SELECT churn_segment,
               COUNT(*) AS customers,
               SUM(total_spent) AS revenue
        FROM churn_risk GROUP BY 1
        """
    )
    mix["churn_segment"] = pd.Categorical(mix["churn_segment"], RISK_ORDER, ordered=True)
    mix = mix.sort_values("churn_segment")
    total = int(mix["customers"].sum())

    kpi_band(
        [
            {
                "label": row.churn_segment,
                "value": f"{int(row.customers):,}",
                "sub": f"{int(row.customers)/total*100:.1f}% of base · {brl(float(row.revenue))}",
                "how": CHURN_SEGMENT_RULES[str(row.churn_segment)]["how"],
                "formula": CHURN_SEGMENT_RULES[str(row.churn_segment)]["formula"],
                "source": "churn_risk",
            }
            for _, row in mix.iterrows()
        ]
    )

    left, right = st.columns(2)
    with left:
        fig = go.Figure(
            go.Pie(
                labels=mix["churn_segment"],
                values=mix["customers"],
                hole=0.62,
                marker=dict(colors=[RISK_COLORS[s] for s in mix["churn_segment"]]),
                textinfo="percent",
                hovertemplate="%{label}<br>%{value:,} buyers<extra></extra>",
            )
        )
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", y=-0.12, x=0),
            margin=dict(l=16, r=16, t=16, b=56),
        )
        chart_card(
            "Risk mix",
            fig,
            how=(
                "Share of buyers in each inactivity band. High Risk is the working definition of churn here — "
                "Olist has no cancel flag. Clock = last order date in the extract, not today’s calendar."
            ),
            formula=(
                "days_inactive = analysis_date − last_order\n"
                "High Risk: days_inactive > 180\n"
                "Medium Risk: days_inactive > 90 AND ≤ 180\n"
                "Low Risk: days_inactive ≤ 90\n"
                "Churn rate = COUNT(High Risk) / COUNT(buyers)"
            ),
            source="churn_risk",
            kind="Donut chart",
        )
    with right:
        fig = px.bar(
            mix,
            x="churn_segment",
            y="revenue",
            color="churn_segment",
            color_discrete_map=RISK_COLORS,
            category_orders={"churn_segment": RISK_ORDER},
        )
        fig.update_traces(hovertemplate="<b>%{x}</b><br>R$ %{y:,.0f}<extra></extra>")
        fig.update_layout(showlegend=False)
        fig.update_yaxes(tickprefix="R$ ")
        polish_vbar(fig, tickangle=0)
        chart_card(
            "Revenue sitting in each band",
            fig,
            how=(
                "Lifetime merchandise value of buyers in each inactivity band. "
                "High Risk revenue is spend that may not return."
            ),
            formula=(
                "Revenue_risk = Σ total_spent  GROUP BY churn_segment\n"
                "High Risk: days_inactive > 180\n"
                "Medium Risk: 90 < days_inactive ≤ 180\n"
                "Low Risk: days_inactive ≤ 90"
            ),
            source="churn_risk.total_spent",
            kind="Vertical bar chart",
        )

    hist = read_sql("SELECT days_inactive FROM churn_risk")
    fig = px.histogram(hist, x="days_inactive", nbins=40)
    fig.update_traces(marker_color=TEAL, hovertemplate="%{x} days<br>%{y:,} buyers<extra></extra>")
    fig.update_layout(showlegend=False, bargap=0.05, margin=dict(l=56, r=16, t=16, b=56))
    fig.update_xaxes(title_text="Days inactive", automargin=True)
    fig.update_yaxes(title_text="Buyers", automargin=True)
    chart_card(
        "Days inactive",
        fig,
        how=(
            "How long since each buyer’s last order, measured from 2018-09-03 (last order in the extract). "
            "Bands cut at 90 and 180 days for Medium / High Risk."
        ),
        formula=(
            "days_inactive = analysis_date − last_order\n"
            "analysis_date = MAX(order_date) = 2018-09-03\n"
            "High Risk > 180 · Medium Risk > 90 · else Low Risk"
        ),
        source="churn_risk ← customer_360_view, analysis_params",
        kind="Histogram",
        height=300,
    )

    tenure = read_sql(
        """
        SELECT EXTRACT(YEAR FROM first_purchase)::int AS cohort_year,
               churn_segment,
               COUNT(*) AS customers
        FROM churn_risk cr
        JOIN customer_metrics m USING (customer_id)
        GROUP BY 1, 2
        ORDER BY 1
        """
    )
    fig = px.bar(
        tenure,
        x="cohort_year",
        y="customers",
        color="churn_segment",
        color_discrete_map=RISK_COLORS,
        category_orders={"churn_segment": RISK_ORDER},
        barmode="stack",
    )
    fig.update_traces(hovertemplate="<b>%{x}</b><br>%{fullData.name}<br>%{y:,} buyers<extra></extra>")
    fig.update_layout(
        legend=dict(orientation="h", y=-0.2, x=0, title_text=""),
        margin=dict(l=56, r=16, t=16, b=72),
    )
    fig.update_xaxes(title_text="Acquisition year", dtick=1, automargin=True)
    fig.update_yaxes(title_text="Buyers", automargin=True)
    chart_card(
        "Risk by acquisition year",
        fig,
        how=(
            "When the buyer first purchased, vs current inactivity band. "
            "Older acquisition years skew High Risk because the analysis clock is fixed at 2018-09-03."
        ),
        formula=(
            "acquisition_year = YEAR(MIN(order_date))\n"
            "COUNT(*) GROUP BY acquisition_year, churn_segment\n"
            "churn_segment: High > 180 days · Medium > 90 · else Low"
        ),
        source="churn_risk ⋈ customer_metrics",
        kind="Stacked bar chart",
    )

    table_card(
        "Highest-value high-risk buyers",
        how=(
            "Buyers who meet the High Risk rule (inactive > 180 days), ranked by lifetime spend. "
            "Includes Recency–Frequency–Monetary frequency and recency for targeting."
        ),
        formula=(
            "Filter: churn_segment = High Risk  (days_inactive > 180)\n"
            "Order by: monetary DESC\n"
            "monetary = Σ amount for that buyer"
        ),
        source="churn_risk ⋈ rfm_segmentation",
        kind="Data table",
    )
    top = read_sql(
        """
        SELECT cr.customer_id,
               cr.days_inactive,
               r.frequency,
               ROUND(r.monetary::numeric, 2) AS monetary,
               r.customer_segment AS rfm_segment
        FROM churn_risk cr
        JOIN rfm_segmentation r USING (customer_id)
        WHERE cr.churn_segment = 'High Risk'
        ORDER BY r.monetary DESC
        LIMIT 40
        """
    )
    st.dataframe(top, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# 4. Cohort retention
# ---------------------------------------------------------------------------

def page_cohort() -> None:
    masthead(
        "Who came back",
        "04  /  Cohort retention",
        ["Cohort = month of first order", "Period 0 is the acquisition month"],
    )
    lede(
        "Olist has no ad-spend table. Retention by first-purchase month is the "
        "honest substitute: it shows whether new buyers ever return."
    )

    repeat = read_sql(
        """
        SELECT
            COUNT(*) FILTER (WHERE total_orders >= 2)::numeric / COUNT(*) AS repeat_rate,
            COUNT(*) FILTER (WHERE total_orders >= 2) AS repeaters,
            COUNT(*) AS buyers
        FROM customer_metrics
        """
    ).iloc[0]
    sizes = read_sql(
        """
        SELECT DISTINCT ON (cohort_month) cohort_month, cohort_size
        FROM cohort_retention
        ORDER BY cohort_month, period_number
        """
    )

    kpi_band(
        [
            {
                "label": "Repeat rate",
                "value": f"{float(repeat.repeat_rate)*100:.1f}%",
                "sub": f"{int(repeat.repeaters):,} of {int(repeat.buyers):,} buyers",
                "how": "Share of people with at least two distinct orders. Olist’s true repeat rate is low; most shoppers buy once.",
                "formula": "RepeatRate = COUNT(orders ≥ 2) / COUNT(buyers)",
                "source": "customer_metrics",
            },
            {
                "label": "Cohorts",
                "value": f"{len(sizes)}",
                "sub": "first-order months",
                "how": "Number of acquisition months with at least one kept order.",
                "formula": "cohort_month = DATE_TRUNC('month', MIN(order_date))",
                "source": "cohort_retention",
            },
            {
                "label": "Largest cohort",
                "value": f"{int(sizes['cohort_size'].max()):,}",
                "sub": pd.to_datetime(sizes.loc[sizes["cohort_size"].idxmax(), "cohort_month"]).strftime("%b %Y"),
                "how": "Headcount of the biggest acquisition month.",
                "formula": "cohort_size = COUNT(*) GROUP BY cohort_month",
                "source": "cohort_retention",
            },
        ]
    )

    cr = read_sql("SELECT * FROM cohort_retention WHERE period_number <= 12")
    cr["cohort_month"] = pd.to_datetime(cr["cohort_month"])
    heat = cr.pivot(index="cohort_month", columns="period_number", values="retention_rate")
    heat = heat.sort_index()
    fig = go.Figure(
        go.Heatmap(
            z=heat.values * 100,
            x=[f"Month {int(c)}" for c in heat.columns],
            y=[d.strftime("%b %Y") for d in heat.index],
            colorscale=[[0, "#F7F3EC"], [0.35, "#D9A066"], [1, TEAL]],
            zmin=0,
            zmax=8,
            hovertemplate="Cohort %{y}<br>%{x}<br>%{z:.2f}% retained<extra></extra>",
            colorbar=dict(title="%"),
        )
    )
    fig.update_layout(margin=dict(l=88, r=24, t=16, b=56))
    fig.update_xaxes(title_text="Months since first order", automargin=True, tickangle=-30)
    fig.update_yaxes(title_text="Cohort", automargin=True)
    chart_card(
        "Retention heatmap",
        fig,
        how="Rows are acquisition months. Columns are months since first order. Cell is the percent of that cohort who placed an order in that later month. Colour scale is capped at 8% so the faint repeat signal is visible.",
        formula="retention(c, p) = buyers in cohort c active in period p\n                         / buyers in cohort c",
        source="cohort_retention",
        kind="Heatmap",
        height=480,
    )

    left, right = st.columns(2)
    with left:
        fig = px.bar(sizes, x="cohort_month", y="cohort_size")
        fig.update_traces(marker_color=TEAL, hovertemplate="%{x|%b %Y}<br>%{y:,} buyers<extra></extra>")
        fig.update_layout(showlegend=False, margin=dict(l=56, r=16, t=16, b=56))
        fig.update_xaxes(title_text="", tickformat="%b %Y", automargin=True)
        fig.update_yaxes(title_text="New buyers", automargin=True)
        chart_card(
            "Cohort size",
            fig,
            how="How many new buyers arrived each month (first kept order).",
            formula="cohort_size = COUNT(*) GROUP BY DATE_TRUNC('month', first_order)",
            source="cohort_retention",
            kind="Vertical bar chart",
        )
    with right:
        aged = read_sql(
            """
            WITH cust_cohort AS (
                SELECT customer_id, DATE_TRUNC('month', MIN(order_date))::date AS cohort_month
                FROM orders_fact GROUP BY customer_id
            )
            SELECT
                ((EXTRACT(YEAR  FROM f.order_date) - EXTRACT(YEAR  FROM c.cohort_month)) * 12
               + (EXTRACT(MONTH FROM f.order_date) - EXTRACT(MONTH FROM c.cohort_month)))::int AS period_number,
                SUM(f.amount) AS revenue
            FROM orders_fact f
            JOIN cust_cohort c USING (customer_id)
            WHERE ((EXTRACT(YEAR  FROM f.order_date) - EXTRACT(YEAR  FROM c.cohort_month)) * 12
                 + (EXTRACT(MONTH FROM f.order_date) - EXTRACT(MONTH FROM c.cohort_month))) BETWEEN 0 AND 12
            GROUP BY 1
            ORDER BY 1
            """
        )
        fig = px.line(aged, x="period_number", y="revenue", markers=True)
        fig.update_traces(
            line_color=TEAL,
            hovertemplate="Month %{x}<br>R$ %{y:,.0f}<extra></extra>",
        )
        fig.update_layout(showlegend=False, margin=dict(l=56, r=16, t=16, b=56))
        fig.update_yaxes(tickprefix="R$ ", title_text="Revenue", automargin=True)
        fig.update_xaxes(title_text="Months since first order", dtick=1, automargin=True)
        chart_card(
            "Revenue by cohort age",
            fig,
            how="All cohorts stacked: GMV booked in the nth month after a buyer’s first order. Period 0 is acquisition spend; later periods are repeat spend.",
            formula="Revenue_p = Σ amount WHERE months_since_first_order = p",
            source="orders_fact + first-order month",
            kind="Line chart",
        )


# ---------------------------------------------------------------------------
# 5. Product & affinity
# ---------------------------------------------------------------------------

def page_product() -> None:
    masthead(
        "What sold together",
        "05  /  Product & affinity",
        ["English category names", "Baskets = order_id"],
    )
    lede(
        "Category mix, the products that carry the book, and association rules "
        "on items that share an order."
    )

    cats = read_sql("SELECT * FROM category_performance ORDER BY revenue DESC")
    kpi_band(
        [
            {
                "label": "Categories",
                "value": f"{len(cats)}",
                "sub": "with at least one sale",
                "how": "Distinct English product categories present on kept order lines.",
                "formula": "COUNT(DISTINCT category)",
                "source": "category_performance",
            },
            {
                "label": "Top category",
                "value": str(cats.iloc[0]["category"]),
                "sub": brl(float(cats.iloc[0]["revenue"])),
                "how": "Category with the highest GMV (price + freight).",
                "formula": "argmax_category Σ amount",
                "source": "category_performance",
            },
            {
                "label": "Category pairs",
                "value": str(int(read_sql("SELECT COUNT(*) AS n FROM category_pairs").iloc[0]["n"])),
                "sub": "rules with ≥ 5 co-orders",
                "how": "How many category pairs appear together in at least five baskets.",
                "formula": "pair_orders ≥ 5",
                "source": "category_pairs",
            },
        ]
    )

    left, right = st.columns(2)
    with left:
        fig = px.treemap(
            cats,
            path=["category"],
            values="revenue",
            color="revenue",
            color_continuous_scale=[[0, "#E6EEEF"], [1, TEAL]],
        )
        fig.update_traces(
            textinfo="label+percent entry",
            hovertemplate=(
                "<b>%{label}</b> · full category<br>"
                "R$ %{value:,.0f} (all SKUs in this category)<extra></extra>"
            ),
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=8, l=0, r=0, b=0))
        chart_card(
            "Revenue by category",
            fig,
            how="Share of GMV by English product category (every SKU in that category summed). Area ∝ category revenue — not a single product.",
            formula="Revenue_cat = Σ amount  GROUP BY category",
            source="category_performance",
            kind="Treemap",
            height=380,
        )
    with right:
        top_p = read_sql(
            """
            SELECT p.product_id, p.category, SUM(f.amount) AS revenue, SUM(f.quantity) AS units
            FROM orders_fact f JOIN dim_products p USING (product_id)
            GROUP BY 1, 2
            ORDER BY revenue DESC
            LIMIT 15
            """
        )
        # Olist has no SKU titles — readable name = rank + English category; full id only on hover.
        top_p = product_display_names(top_p)
        top_p = top_p.sort_values("revenue", ascending=True)
        fig = px.bar(
            top_p,
            x="revenue",
            y="product_name",
            orientation="h",
            color="category",
            custom_data=["product_id", "units"],
        )
        fig.update_traces(
            hovertemplate=(
                "<b>%{y}</b> · one SKU<br>"
                "product_id %{customdata[0]}<br>"
                "R$ %{x:,.0f} · %{customdata[1]:,.0f} units<br>"
                "<i>Not the full category total (see treemap)</i><extra></extra>"
            )
        )
        fig.update_xaxes(tickprefix="R$ ")
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=list(top_p["product_name"]),
            tickfont=dict(size=11),
        )
        polish_hbar(fig, left=168)
        fig.update_layout(
            legend=dict(orientation="h", y=-0.22, x=0, title_text=""),
            margin=dict(l=168, r=16, t=12, b=72),
        )
        chart_card(
            "Top 15 products (single SKUs)",
            fig,
            how="Highest-GMV individual products. Label is GMV-rank + category because Olist has no product titles. Hover shows product_id. These amounts are one SKU each — far smaller than the category totals on the treemap (e.g. top Health Beauty SKU ≈ R$ 68k vs category ≈ R$ 1.44M).",
            formula="Revenue_p = Σ amount  GROUP BY product_id",
            source="orders_fact ⋈ dim_products",
            kind="Horizontal bar chart",
            height=420,
        )

    # Label top categories on the scatter so points stay readable without a huge legend.
    top_labels = set(cats.head(8)["category"])
    cats_plot = cats.copy()
    cats_plot["label"] = cats_plot["category"].where(cats_plot["category"].isin(top_labels), "")
    fig = px.scatter(
        cats_plot,
        x="units",
        y="revenue",
        size="orders",
        color="category",
        text="label",
        hover_name="category",
        size_max=42,
    )
    fig.update_traces(
        textposition="top center",
        textfont=dict(size=10),
        hovertemplate="<b>%{hovertext}</b><br>%{x:,} units<br>R$ %{y:,.0f}<extra></extra>",
    )
    fig.update_xaxes(title_text="Units", automargin=True)
    fig.update_yaxes(tickprefix="R$ ", title_text="Revenue", automargin=True)
    fig.update_layout(
        showlegend=False,
        margin=dict(l=64, r=24, t=24, b=48),
    )
    chart_card(
        "Units vs revenue",
        fig,
        how="Each point is a category. Size is distinct orders. Top categories are labeled; hover any point for the name.",
        formula="x = Σ quantity    y = Σ amount    size = COUNT DISTINCT order_id",
        source="category_performance",
        kind="Bubble scatter plot",
        height=380,
    )

    share = read_sql(
        """
        WITH monthly AS (
            SELECT DATE_TRUNC('month', f.order_date)::date AS month,
                   p.category,
                   SUM(f.amount) AS revenue
            FROM orders_fact f JOIN dim_products p USING (product_id)
            GROUP BY 1, 2
        ),
        top AS (
            SELECT category FROM category_performance ORDER BY revenue DESC LIMIT 6
        )
        SELECT m.month, COALESCE(t.category, 'Other') AS category, SUM(m.revenue) AS revenue
        FROM monthly m
        LEFT JOIN top t ON t.category = m.category
        GROUP BY 1, 2
        ORDER BY 1
        """
    )
    share["month"] = pd.to_datetime(share["month"])
    fig = px.area(share, x="month", y="revenue", color="category", groupnorm="percent")
    fig.update_traces(hovertemplate="<b>%{fullData.name}</b><br>%{x|%b %Y}<br>%{y:.1f}%<extra></extra>")
    fig.update_yaxes(ticksuffix="%", title_text="Share of month", automargin=True)
    fig.update_xaxes(title_text="", automargin=True)
    fig.update_layout(
        legend=dict(orientation="h", y=-0.28, x=0, title_text=""),
        margin=dict(l=56, r=16, t=16, b=96),
    )
    chart_card(
        "Category share over time",
        fig,
        how="Top six categories by total GMV, plus Other. Stack is 100% of that month’s revenue.",
        formula="share_cat,m = Revenue_cat,m / Σ_cat Revenue_m",
        source="orders_fact ⋈ dim_products",
        kind="Stacked area chart",
        height=400,
    )

    table_card(
        "Category association rules",
        how="Market basket at category grain. Basket = order. Lift > 1 means the pair is bought together more than chance; most Olist baskets are single-category, so lift often sits below 1.",
        formula="Support(A,B) = orders with A and B / all orders\nConfidence(A→B) = orders with A and B / orders with A\nLift(A,B) = Support(A,B) / (Support(A)×Support(B))",
        source="category_pairs",
        kind="Data table",
    )
    pairs = read_sql(
        """
        SELECT category_a, category_b, pair_orders, support, confidence_a_to_b, lift
        FROM category_pairs
        ORDER BY lift DESC, pair_orders DESC
        LIMIT 20
        """
    )
    st.dataframe(pairs, width="stretch", hide_index=True)

    table_card(
        "Product pairs (min. 3 co-orders)",
        how="Same rules at product grain. Olist has no product titles, so rows show category names with a short id suffix. Baskets are sparse here — short, high-signal list.",
        formula="Support / Confidence / Lift on (product_a, product_b) where pair_orders ≥ 3",
        source="product_pairs",
        kind="Data table",
    )
    pp = read_sql(
        """
        SELECT product_a, category_a, product_b, category_b, pair_orders, lift, confidence_a_to_b
        FROM product_pairs
        ORDER BY lift DESC, pair_orders DESC
        LIMIT 20
        """
    )
    pp_view = pd.DataFrame(
        {
            "product_a": pp["category_a"].astype(str) + " · " + pp["product_a"].astype(str).str[:8],
            "product_b": pp["category_b"].astype(str) + " · " + pp["product_b"].astype(str).str[:8],
            "pair_orders": pp["pair_orders"],
            "lift": pp["lift"],
            "confidence_a_to_b": pp["confidence_a_to_b"],
        }
    )
    st.dataframe(pp_view, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="UrbanCart · Olist",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_theme()

    try:
        params = read_sql("SELECT * FROM analysis_params").iloc[0]
    except Exception as exc:
        st.error(
            "Database is not ready. From this folder run:\n\n"
            "`python transform_olist.py`  then  `python setup_local.py`\n\n"
            f"{exc}"
        )
        st.stop()

    tabs = st.tabs(
        [
            "01  Executive",
            "02  RFM",
            "03  Churn",
            "04  Cohorts",
            "05  Product",
        ]
    )
    with tabs[0]:
        page_executive()
    with tabs[1]:
        page_rfm()
    with tabs[2]:
        page_churn()
    with tabs[3]:
        page_cohort()
    with tabs[4]:
        page_product()

    footer(
        "UrbanCart  ·  analysis date "
        + pd.to_datetime(params["analysis_date"]).strftime("%d %b %Y")
        + "  ·  Power BI: powerbi/UrbanCart_Olist.pbip",
        "Source: Olist Brazilian E-Commerce  ·  CC BY-NC-SA 4.0",
    )


if __name__ == "__main__":
    main()
