"""
Dash application factory.

create_app() builds and returns the configured Dash app.
Keeping app creation in a factory function (rather than at module level)
makes testing easier and avoids circular import issues.

Run from the repo root:
    conda activate jse-radar
    python -m jse_radar.dashboard.app
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc

from jse_radar.dashboard.data_loader import DashboardData
from jse_radar.dashboard.layout import make_layout
from jse_radar.dashboard.callbacks import register_callbacks
from jse_radar.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> dash.Dash:
    """
    Load data, build layout, register callbacks, return app.
    """
    # ── Load all data once at startup ─────────────────────────────────────────
    data = DashboardData()

    # ── Create the Dash app ───────────────────────────────────────────────────
    # We use the DARKLY Bootstrap theme for base styling.
    # Dash Bootstrap Components (dbc) gives us a grid system and
    # pre-styled components that work with the theme.
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="jse-radar",
        suppress_callback_exceptions=True,
    )

    # ── Set layout ────────────────────────────────────────────────────────────
    app.layout = make_layout(
        tickers=data.tickers,
        ticker_names=data.ticker_names,
        macro_vars=data.macro_vars,
        date_min=data.date_min,
        date_max=data.date_max,
    )

    # ── Register all callbacks ────────────────────────────────────────────────
    register_callbacks(app, data)

    return app


if __name__ == "__main__":
    app = create_app()
    logger.info("Starting jse-radar dashboard on http://127.0.0.1:8050")
    app.run(debug=True, port=8050)