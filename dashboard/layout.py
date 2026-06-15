"""
Dashboard layout — defines the visual structure of the app.

Dash layouts are built from Python objects that map to HTML/CSS.
html.Div is a <div>, dcc.Graph is a Plotly chart, dcc.Dropdown
is a dropdown menu. They nest exactly like HTML elements.

We split layout into a separate file from callbacks so that
as the app grows, neither file becomes unmanageable.
"""

from __future__ import annotations

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc


# ── Colour palette — consistent across all charts ────────────────────────────
COLORS = {
    "background":  "#0d1117",
    "surface":     "#161b22",
    "border":      "#30363d",
    "text":        "#e6edf3",
    "text_muted":  "#8b949e",
    "accent":      "#58a6ff",
    "positive":    "#3fb950",
    "negative":    "#f85149",
    "warning":     "#d29922",
}

PLOTLY_TEMPLATE = "plotly_dark"

# ── Reusable style dicts ──────────────────────────────────────────────────────
CARD_STYLE = {
    "backgroundColor": COLORS["surface"],
    "border":          f"1px solid {COLORS['border']}",
    "borderRadius":    "8px",
    "padding":         "16px",
    "marginBottom":    "16px",
}

LABEL_STYLE = {
    "color":        COLORS["text_muted"],
    "fontSize":     "12px",
    "marginBottom": "4px",
    "fontWeight":   "500",
    "letterSpacing": "0.05em",
    "textTransform": "uppercase",
}


def make_layout(tickers: list[str], ticker_names: dict, macro_vars: list[str],
                date_min, date_max) -> html.Div:
    """
    Build and return the full app layout.
    Called once at startup with data from DashboardData.
    """

    # Dropdown options: show "NPN.JO — Naspers" instead of just the ticker
    ticker_options = [
        {
            "label": f"{t}  —  {ticker_names.get(t, t)}",
            "value": t,
        }
        for t in tickers
    ]

    macro_labels = {
        "zar_usd_mom_pct":  "ZAR/USD Monthly Change (%)",
        "tbill_rate":        "T-Bill Rate / Repo Proxy (%)",
        "cpi_yoy_pct":       "CPI Year-on-Year (%)",
        "real_tbill_rate":   "Real Interest Rate (%)",
        "exports_value":     "Exports Value (ZAR)",
        "imports_value":     "Imports Value (ZAR)",
    }
    macro_options = [
        {"label": macro_labels.get(v, v), "value": v}
        for v in macro_vars
    ]

    return html.Div(
        style={"backgroundColor": COLORS["background"], "minHeight": "100vh",
               "fontFamily": "'Inter', 'Segoe UI', sans-serif",
               "color": COLORS["text"]},
        children=[

            # ── Header ────────────────────────────────────────────────────────
            html.Div(
                style={"backgroundColor": COLORS["surface"],
                       "borderBottom": f"1px solid {COLORS['border']}",
                       "padding": "16px 32px",
                       "marginBottom": "24px"},
                children=[
                    html.H1("jse-radar",
                            style={"margin": 0, "fontSize": "24px",
                                   "color": COLORS["accent"],
                                   "fontWeight": "700",
                                   "letterSpacing": "-0.02em"}),
                    html.P("JSE Macroeconomic Data Pipeline & Dashboard",
                           style={"margin": "4px 0 0 0",
                                  "color": COLORS["text_muted"],
                                  "fontSize": "13px"}),
                ]
            ),

            # ── Tabs ──────────────────────────────────────────────────────────
            html.Div(
                style={"padding": "0 32px 32px 32px"},
                children=[
                    dcc.Tabs(
                        id="tabs",
                        value="tab-market",
                        style={"marginBottom": "24px"},
                        colors={
                            "border":     COLORS["border"],
                            "primary":    COLORS["accent"],
                            "background": COLORS["surface"],
                        },
                        children=[

                            # ════════════════════════════════════════════════
                            # TAB 1 — MARKET OVERVIEW
                            # ════════════════════════════════════════════════
                            dcc.Tab(
                                label="Market Overview",
                                value="tab-market",
                                children=[

                                    # Controls row
                                    html.Div(
                                        style={**CARD_STYLE,
                                               "display": "flex",
                                               "gap": "24px",
                                               "flexWrap": "wrap",
                                               "alignItems": "flex-end"},
                                        children=[
                                            html.Div([
                                                html.Label("Select Ticker",
                                                           style=LABEL_STYLE),
                                                dcc.Dropdown(
                                                    id="market-ticker-dropdown",
                                                    options=ticker_options,
                                                    value=tickers[0] if tickers else None,
                                                    style={"width": "320px",
                                                           "backgroundColor": COLORS["surface"],
                                                           "color": COLORS["text"]},
                                                    clearable=False,
                                                ),
                                            ]),
                                            html.Div([
                                                html.Label("Date Range",
                                                           style=LABEL_STYLE),
                                                dcc.DatePickerRange(
                                                    id="market-date-range",
                                                    min_date_allowed=date_min,
                                                    max_date_allowed=date_max,
                                                    start_date=date_min,
                                                    end_date=date_max,
                                                    display_format="YYYY-MM-DD",
                                                    style={"fontSize": "13px"},
                                                ),
                                            ]),
                                        ]
                                    ),

                                    # Price + volume chart
                                    html.Div(style=CARD_STYLE, children=[
                                        dcc.Graph(id="market-price-chart",
                                                  style={"height": "420px"}),
                                    ]),

                                    # RSI chart
                                    html.Div(style=CARD_STYLE, children=[
                                        dcc.Graph(id="market-rsi-chart",
                                                  style={"height": "220px"}),
                                    ]),
                                ]
                            ),

                            # ════════════════════════════════════════════════
                            # TAB 2 — MACRO ENVIRONMENT
                            # ════════════════════════════════════════════════
                            dcc.Tab(
                                label="Macro Environment",
                                value="tab-macro",
                                children=[

                                    # Current regime banner
                                    html.Div(
                                        id="regime-banner",
                                        style={**CARD_STYLE, "textAlign": "center"},
                                    ),

                                    # Four macro charts
                                    html.Div(style=CARD_STYLE, children=[
                                        dcc.Graph(id="macro-overview-chart",
                                                  style={"height": "600px"}),
                                    ]),

                                    # Regime history timeline
                                    html.Div(style=CARD_STYLE, children=[
                                        dcc.Graph(id="regime-timeline-chart",
                                                  style={"height": "300px"}),
                                    ]),
                                ]
                            ),

                            # ════════════════════════════════════════════════
                            # TAB 3 — SIGNALS
                            # ════════════════════════════════════════════════
                            dcc.Tab(
                                label="Signals",
                                value="tab-signals",
                                children=[

                                    # Date selector for signals snapshot
                                    html.Div(
                                        style={**CARD_STYLE,
                                               "display": "flex",
                                               "gap": "24px",
                                               "alignItems": "flex-end"},
                                        children=[
                                            html.Div([
                                                html.Label(
                                                    "Snapshot Date (latest available)",
                                                    style=LABEL_STYLE),
                                                dcc.DatePickerSingle(
                                                    id="signals-date-picker",
                                                    min_date_allowed=date_min,
                                                    max_date_allowed=date_max,
                                                    date=date_max,
                                                    display_format="YYYY-MM-DD",
                                                ),
                                            ]),
                                        ]
                                    ),

                                    # Momentum score bar chart
                                    html.Div(style=CARD_STYLE, children=[
                                        dcc.Graph(id="signals-momentum-chart",
                                                  style={"height": "420px"}),
                                    ]),

                                    # Signals table
                                    html.Div(style=CARD_STYLE, children=[
                                        html.Label("Signal Snapshot — All Tickers",
                                                   style={**LABEL_STYLE,
                                                          "marginBottom": "12px"}),
                                        html.Div(id="signals-table"),
                                    ]),
                                ]
                            ),

                            # ════════════════════════════════════════════════
                            # TAB 4 — CORRELATIONS
                            # ════════════════════════════════════════════════
                            dcc.Tab(
                                label="Correlations",
                                value="tab-correlations",
                                children=[

                                    # Controls
                                    html.Div(
                                        style={**CARD_STYLE,
                                               "display": "flex",
                                               "gap": "24px",
                                               "flexWrap": "wrap",
                                               "alignItems": "flex-end"},
                                        children=[
                                            html.Div([
                                                html.Label("Select Ticker",
                                                           style=LABEL_STYLE),
                                                dcc.Dropdown(
                                                    id="corr-ticker-dropdown",
                                                    options=ticker_options,
                                                    value=tickers[0] if tickers else None,
                                                    style={"width": "320px"},
                                                    clearable=False,
                                                ),
                                            ]),
                                            html.Div([
                                                html.Label("Macro Variable",
                                                           style=LABEL_STYLE),
                                                dcc.Dropdown(
                                                    id="corr-macro-dropdown",
                                                    options=macro_options,
                                                    value=macro_vars[0] if macro_vars else None,
                                                    style={"width": "320px"},
                                                    clearable=False,
                                                ),
                                            ]),
                                        ]
                                    ),

                                    # Rolling correlation chart
                                    html.Div(style=CARD_STYLE, children=[
                                        dcc.Graph(id="rolling-corr-chart",
                                                  style={"height": "400px"}),
                                    ]),

                                    # Static correlation matrix
                                    html.Div(style=CARD_STYLE, children=[
                                        html.Label(
                                            "Return Correlation Matrix — All Tickers",
                                            style={**LABEL_STYLE,
                                                   "marginBottom": "12px"}),
                                        dcc.Graph(id="corr-matrix-chart",
                                                  style={"height": "600px"}),
                                    ]),
                                ]
                            ),
                        ]
                    )
                ]
            )
        ]
    )