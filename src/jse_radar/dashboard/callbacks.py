"""
Dashboard callbacks — all interactive behaviour lives here.

Every callback follows the same pattern:
  @app.callback(
      Output("component-id", "property"),   # what to update
      Input("component-id",  "property"),   # what triggers it
  )
  def my_function(input_value):
      # compute something
      return new_value_for_output_property

When the Input changes, Dash calls the function automatically
and puts the return value into the Output's property.
Multiple Inputs = function is called when ANY of them changes.
Multiple Outputs = function must return a tuple.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Input, Output, html
from dash.dash_table import DataTable

from jse_radar.dashboard.layout import COLORS, PLOTLY_TEMPLATE


def register_callbacks(app, data):
    """
    Register all callbacks with the Dash app instance.
    Called once at startup after the app and layout are created.
    `data` is a DashboardData instance.
    """

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — MARKET OVERVIEW CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("market-price-chart", "figure"),
        Output("market-rsi-chart",   "figure"),
        Input("market-ticker-dropdown", "value"),
        Input("market-date-range",      "start_date"),
        Input("market-date-range",      "end_date"),
    )
    def update_market_charts(ticker, start_date, end_date):
        """
        Update the price/volume chart and RSI chart when the user
        changes the ticker or date range.

        We use make_subplots to stack two charts vertically:
          - Top: candlestick price with MA50 and MA200
          - Bottom: volume bars
        The RSI chart is a separate figure below.
        """
        if not ticker or data.master is None:
            empty = go.Figure()
            empty.update_layout(template=PLOTLY_TEMPLATE,
                                paper_bgcolor=COLORS["background"])
            return empty, empty

        # Filter to selected ticker and date range
        df = data.master[data.master["ticker"] == ticker].copy()
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

        if df.empty:
            empty = go.Figure()
            return empty, empty

        name = data.ticker_names.get(ticker, ticker)

        # ── Price + volume + MAs ──────────────────────────────────────────────
        fig_price = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.03,
        )

        # Candlestick — shows open/high/low/close for each day
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            fig_price.add_trace(
                go.Candlestick(
                    x=df["date"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name=ticker,
                    increasing_line_color=COLORS["positive"],
                    decreasing_line_color=COLORS["negative"],
                    showlegend=False,
                ),
                row=1, col=1,
            )

        # MA50 overlay
        if "ma_50" in df.columns:
            fig_price.add_trace(
                go.Scatter(
                    x=df["date"], y=df["ma_50"],
                    name="MA50", line={"color": "#f0b429", "width": 1.5},
                    opacity=0.9,
                ),
                row=1, col=1,
            )

        # MA200 overlay
        if "ma_200" in df.columns:
            fig_price.add_trace(
                go.Scatter(
                    x=df["date"], y=df["ma_200"],
                    name="MA200", line={"color": "#9f7aea", "width": 1.5},
                    opacity=0.9,
                ),
                row=1, col=1,
            )

        # Volume bars
        if "volume" in df.columns:
            fig_price.add_trace(
                go.Bar(
                    x=df["date"], y=df["volume"],
                    name="Volume",
                    marker_color=COLORS["accent"],
                    opacity=0.4,
                    showlegend=False,
                ),
                row=2, col=1,
            )

        fig_price.update_layout(
            title=f"{ticker} — {name}",
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["background"],
            xaxis_rangeslider_visible=False,
            legend={"orientation": "h", "y": 1.02},
            margin={"t": 50, "b": 10, "l": 10, "r": 10},
        )
        fig_price.update_yaxes(title_text="Price (ZAR)", row=1, col=1)
        fig_price.update_yaxes(title_text="Volume",      row=2, col=1)

        # ── RSI chart ─────────────────────────────────────────────────────────
        fig_rsi = go.Figure()

        if "rsi_14" in df.columns:
            fig_rsi.add_trace(go.Scatter(
                x=df["date"], y=df["rsi_14"],
                name="RSI 14",
                line={"color": COLORS["accent"], "width": 1.5},
            ))
            # Overbought / oversold reference lines
            for level, colour, label in [
                (70, COLORS["negative"], "Overbought"),
                (30, COLORS["positive"], "Oversold"),
                (50, COLORS["text_muted"], ""),
            ]:
                fig_rsi.add_hline(
                    y=level,
                    line_dash="dash",
                    line_color=colour,
                    opacity=0.6,
                    annotation_text=label,
                    annotation_position="left",
                )

        fig_rsi.update_layout(
            title="RSI (14-day)",
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["background"],
            yaxis={"range": [0, 100]},
            margin={"t": 40, "b": 10, "l": 10, "r": 10},
            showlegend=False,
        )

        return fig_price, fig_rsi

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — MACRO ENVIRONMENT CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("macro-overview-chart",  "figure"),
        Output("regime-banner",         "children"),
        Output("regime-timeline-chart", "figure"),
        Input("tabs", "value"),
    )
    def update_macro_tab(tab):
        """
        Triggered when the user switches to the Macro tab.
        Renders the four macro charts, the current regime banner,
        and the regime timeline.
        """
        if tab != "tab-macro" or data.macro is None:
            empty = go.Figure()
            empty.update_layout(template=PLOTLY_TEMPLATE,
                                paper_bgcolor=COLORS["background"])
            return empty, "", empty

        macro = data.macro.copy()

        # ── Four macro subplots ───────────────────────────────────────────────
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "T-Bill Rate / Repo Proxy (%)",
                "CPI All Items (Index, 2015=100)",
                "ZAR / USD Exchange Rate",
                "ZAR/USD Month-on-Month Change (%)",
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.10,
        )

        plot_series = [
            ("tbill_rate",       1, 1, COLORS["warning"]),
            ("cpi_all",          1, 2, "#f87171"),
            ("zar_usd",          2, 1, "#34d399"),
            ("zar_usd_mom_pct",  2, 2, "#a78bfa"),
        ]

        for col, row, col_pos, colour in plot_series:
            if col in macro.columns:
                fig.add_trace(
                    go.Scatter(
                        x=macro["date"], y=macro[col],
                        name=col, mode="lines",
                        line={"color": colour, "width": 1.8},
                    ),
                    row=row, col=col_pos,
                )

        fig.update_layout(
            title="South African Macro Indicators (FRED)",
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=COLORS["background"],
            plot_bgcolor=COLORS["background"],
            showlegend=False,
            margin={"t": 60, "b": 20, "l": 10, "r": 10},
        )

        # ── Regime banner ─────────────────────────────────────────────────────
        regime_banner = html.P(
            "Regime data not available — run SignalEngine and MacroRegimeClassifier first.",
            style={"color": COLORS["text_muted"]},
        )

        if data.regimes is not None:
            latest = data.regimes.dropna(subset=["composite_regime"]).iloc[-1]
            regime     = latest.get("composite_regime", "UNKNOWN")
            duration   = int(latest.get("regime_duration", 0))
            tbill      = latest.get("tbill_rate", None)
            cpi        = latest.get("cpi_yoy_pct", None)

            # Colour the banner by regime type
            regime_colour = (
                COLORS["negative"] if "HIKING" in str(regime)
                else COLORS["positive"] if "CUTTING" in str(regime)
                else COLORS["warning"]
            )

            regime_banner = html.Div([
                html.H3(
                    f"Current Macro Regime: {regime}",
                    style={"color": regime_colour, "margin": "0 0 8px 0",
                           "fontSize": "20px"}
                ),
                html.P(
                    f"Active for {duration} month(s)  |  "
                    f"T-Bill: {tbill:.2f}%  |  "
                    f"CPI YoY: {cpi:.2f}%"
                    if tbill is not None and cpi is not None
                    else f"Active for {duration} month(s)",
                    style={"color": COLORS["text_muted"], "margin": 0,
                           "fontSize": "14px"},
                ),
            ])

        # ── Regime timeline ───────────────────────────────────────────────────
        fig_regime = go.Figure()

        if data.regimes is not None and "composite_regime" in data.regimes.columns:
            regimes = data.regimes.dropna(subset=["composite_regime"]).copy()

            # Map regime to a numeric value for colour coding
            regime_map = {
                r: i for i, r in enumerate(regimes["composite_regime"].unique())
            }
            regimes["regime_num"] = regimes["composite_regime"].map(regime_map)

            fig_regime.add_trace(go.Scatter(
                x=regimes["date"],
                y=regimes["composite_regime"],
                mode="markers",
                marker={
                    "color":   regimes["regime_num"],
                    "colorscale": "RdYlGn",
                    "size":    8,
                    "showscale": False,
                },
                text=regimes["composite_regime"],
                hovertemplate="%{x|%b %Y}: %{text}<extra></extra>",
            ))

            fig_regime.update_layout(
                title="Macro Regime History",
                template=PLOTLY_TEMPLATE,
                paper_bgcolor=COLORS["background"],
                plot_bgcolor=COLORS["background"],
                margin={"t": 40, "b": 20, "l": 10, "r": 10},
                yaxis={"title": "Regime"},
            )

        return fig, regime_banner, fig_regime

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — SIGNALS CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("signals-momentum-chart", "figure"),
        Output("signals-table",          "children"),
        Input("signals-date-picker",     "date"),
    )
    def update_signals(selected_date):
        """
        Show a snapshot of all signals on or before the selected date.
        Uses asof logic: finds the latest available data for each ticker
        up to the selected date.
        """
        if data.master is None:
            empty = go.Figure()
            return empty, "No data available."

        df = data.master.copy()
        df = df[df["date"] <= pd.Timestamp(selected_date)]

        # Get the most recent row per ticker (latest available signal)
        snapshot = (
            df.sort_values("date")
            .groupby("ticker")
            .last()
            .reset_index()
        )

        signal_cols = [
            "momentum_score", "rsi_14", "trend_signal",
            "mom_1m", "mom_3m", "mom_6m",
        ]
        available_cols = [c for c in signal_cols if c in snapshot.columns]

        if not available_cols:
            empty = go.Figure()
            return empty, "Signal columns not found. Run SignalEngine first."

        # ── Momentum bar chart ────────────────────────────────────────────────
        if "momentum_score" in snapshot.columns:
            snap_sorted = snapshot.sort_values("momentum_score")
            fig = go.Figure(go.Bar(
                x=snap_sorted["ticker"],
                y=snap_sorted["momentum_score"],
                marker_color=np.where(
                    snap_sorted["momentum_score"] > 0,
                    COLORS["positive"],
                    COLORS["negative"],
                ),
                text=snap_sorted["momentum_score"].round(2),
                textposition="outside",
            ))
            fig.update_layout(
                title=f"Composite Momentum Score — {selected_date}",
                template=PLOTLY_TEMPLATE,
                paper_bgcolor=COLORS["background"],
                plot_bgcolor=COLORS["background"],
                xaxis_title="Ticker",
                yaxis_title="Momentum Score (cross-sectional z-score)",
                margin={"t": 50, "b": 60, "l": 10, "r": 10},
            )
        else:
            fig = go.Figure()

        # ── Signals table ─────────────────────────────────────────────────────
        table_cols = ["ticker"] + available_cols
        if "name" in snapshot.columns:
            table_cols = ["ticker", "name"] + available_cols

        table_df = snapshot[
            [c for c in table_cols if c in snapshot.columns]
        ].round(4)

        table = DataTable(
            data=table_df.to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c}
                     for c in table_df.columns],
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": COLORS["surface"],
                "color":           COLORS["text"],
                "fontWeight":      "600",
                "border":          f"1px solid {COLORS['border']}",
                "fontSize":        "12px",
            },
            style_cell={
                "backgroundColor": COLORS["background"],
                "color":           COLORS["text"],
                "border":          f"1px solid {COLORS['border']}",
                "fontSize":        "12px",
                "padding":         "8px 12px",
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": "{trend_signal} = 1"},
                    "backgroundColor": "#0d2818",
                },
                {
                    "if": {"filter_query": "{trend_signal} = -1"},
                    "backgroundColor": "#2d0f0f",
                },
            ],
        )

        return fig, table

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — CORRELATIONS CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("rolling-corr-chart", "figure"),
        Output("corr-matrix-chart",  "figure"),
        Input("corr-ticker-dropdown", "value"),
        Input("corr-macro-dropdown",  "value"),
    )
    def update_correlations(ticker, macro_var):
        """
        Rolling correlation chart for the selected ticker vs macro variable.
        Static return correlation matrix for all tickers.
        """
        empty = go.Figure()
        empty.update_layout(template=PLOTLY_TEMPLATE,
                            paper_bgcolor=COLORS["background"])

        # ── Rolling correlation ───────────────────────────────────────────────
        fig_rolling = empty

        if (data.correlations is not None
                and ticker and macro_var):
            corr = data.correlations[
                (data.correlations["ticker"]    == ticker) &
                (data.correlations["macro_var"] == macro_var)
            ].copy()

            if not corr.empty:
                name = data.ticker_names.get(ticker, ticker)
                fig_rolling = go.Figure()

                # Zero reference line
                fig_rolling.add_hline(
                    y=0, line_dash="dash",
                    line_color=COLORS["text_muted"], opacity=0.5,
                )

                # Fill above/below zero with green/red
                fig_rolling.add_trace(go.Scatter(
                    x=corr["date"],
                    y=corr["rolling_corr"],
                    mode="lines",
                    name=f"90d rolling corr",
                    line={"color": COLORS["accent"], "width": 1.5},
                    fill="tozeroy",
                    fillcolor="rgba(88, 166, 255, 0.15)",
                ))

                fig_rolling.update_layout(
                    title=f"90-Day Rolling Correlation: {ticker} ({name}) vs {macro_var}",
                    template=PLOTLY_TEMPLATE,
                    paper_bgcolor=COLORS["background"],
                    plot_bgcolor=COLORS["background"],
                    yaxis={"range": [-1, 1], "title": "Correlation"},
                    xaxis_title="Date",
                    margin={"t": 50, "b": 30, "l": 10, "r": 10},
                    showlegend=False,
                )

        # ── Static correlation matrix ─────────────────────────────────────────
        fig_matrix = empty

        if data.master is not None:
            # Pivot to wide: date × ticker → daily_return
            wide = data.master.pivot_table(
                index="date",
                columns="ticker",
                values="daily_return",
            )
            corr_matrix = wide.corr().round(3)

            fig_matrix = go.Figure(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=corr_matrix.values.round(2),
                texttemplate="%{text}",
                textfont={"size": 9},
                hoverongaps=False,
            ))

            fig_matrix.update_layout(
                title="Return Correlation Matrix — All JSE Tickers",
                template=PLOTLY_TEMPLATE,
                paper_bgcolor=COLORS["background"],
                plot_bgcolor=COLORS["background"],
                margin={"t": 50, "b": 10, "l": 10, "r": 10},
            )

        return fig_rolling, fig_matrix