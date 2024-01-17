import dash
from dash import Output, Input, State, ctx
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import callback
import logging
from RAPDOR.plots import COLOR_SCHEMES

logger = logging.getLogger(__name__)

@callback(
    [
        Output("primary-2-color-modal", "is_open"),
        Output("primary-color", "data", allow_duplicate=True),
        Output("color-scheme-dropdown", "value", allow_duplicate=True),

    ],
    [
        Input("primary-2-open-color-modal", "n_clicks"),
        Input("primary-2-apply-color-modal", "n_clicks"),
    ],
    [
        State("primary-2-color-modal", "is_open"),
        State("primary-2-color-picker", "value"),
        State("primary-2-open-color-modal", "style"),

    ],
    prevent_initial_call=True
)
def _toggle_primary_color_modal(n1, n2, is_open, color_value, style):
    logger.info(f"{ctx.triggered_id} - triggered secondary color modal")
    tid = ctx.triggered_id
    if n1 == 0:
        raise PreventUpdate
    if tid == "primary-2-open-color-modal":
        return not is_open, dash.no_update, dash.no_update
    elif tid == "primary-2-apply-color-modal":
        rgb = color_value["rgb"]
        r, g, b = rgb["r"], rgb["g"], rgb["b"]
        color = f"rgb({r}, {g}, {b})"
    else:
        raise ValueError("")
    return not is_open, color, None


@callback(
    [
        Output("secondary-2-color-modal", "is_open"),
        Output("secondary-color", "data", allow_duplicate=True),
        Output("color-scheme-dropdown", "value", allow_duplicate=True),

    ],
    [
        Input("secondary-2-open-color-modal", "n_clicks"),
        Input("secondary-2-apply-color-modal", "n_clicks"),
    ],
    [
        State("secondary-2-color-modal", "is_open"),
        State("secondary-2-color-picker", "value"),
        State("secondary-2-open-color-modal", "style"),

    ],
    prevent_initial_call=True
)
def _toggle_secondary_color_modal(n1, n2, is_open, color_value, style):
    logger.info(f"{ctx.triggered_id} - triggered secondary color modal")
    tid = ctx.triggered_id
    if n1 == 0:
        raise PreventUpdate
    if tid == "secondary-2-open-color-modal":
        return not is_open, dash.no_update, dash.no_update
    elif tid == "secondary-2-apply-color-modal":
        rgb = color_value["rgb"]
        r, g, b = rgb["r"], rgb["g"], rgb["b"]
        color = f"rgb({r}, {g}, {b})"
    else:
        raise ValueError("")
    return not is_open, color, None


@callback(
    Output("primary-color", "data", allow_duplicate=True),
    Output("secondary-color", "data", allow_duplicate=True),
    Input("color-scheme-dropdown", "value"),

    prevent_initital_call=True
)
def _open_color_theme_modal(color_scheme):
    if color_scheme is not None:
        primary, secondary = COLOR_SCHEMES[color_scheme]
    else:
        raise PreventUpdate

    return primary, secondary
