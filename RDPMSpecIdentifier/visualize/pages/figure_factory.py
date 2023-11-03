
import dash
from dash import html
import base64
from dash import dcc
from dash_extensions.enrich import callback, Input, Output, Serverside, State, ctx
from RDPMSpecIdentifier.visualize.staticContent import COLOR_SCHEMES
from RDPMSpecIdentifier.datastructures import RDPMSpecData
import logging
import dash_bootstrap_components as dbc
import pandas as pd
from io import StringIO
from dash.exceptions import PreventUpdate
import os
from RDPMSpecIdentifier.visualize import DISABLED
from tempfile import NamedTemporaryFile
from RDPMSpecIdentifier.plots import plot_protein_distributions
import numpy as np
from RDPMSpecIdentifier.visualize.callbacks.modalCallbacks import FILEEXT
from RDPMSpecIdentifier.visualize.modals import _color_theme_modal, _modal_color_selection
from io import BytesIO
import plotly.io as pio
dash.register_page(__name__, path='/figure_factory')

logger = logging.getLogger(__name__)


def _args_and_name(input_id, arg, d_type, default):
    if isinstance(default, int):
        step = 1
    elif isinstance(default, float):
        step = 0.01
    else:
        step = None
    div = [
            html.Div(
                html.Span(arg, style={"text-align": "center"}),
                className="col-2 justify-content-center align-self-center py-1"
            ),
            html.Div(
                dcc.Input(
                    style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white",
                           "text-align": "center"},
                    id=input_id,
                    className="text-align-center",
                    value=default,
                    type=d_type,
                    step=step,
                    persistence=True,
                    persistence_type="session"
                ),
                className="col-8 col-md-4 justify-content-center text-align-center py-1"
            )
        ]
    return div

def _arg_and_dropdown(arg, dd_list, default, input_id):
    div = [
        html.Div(
            html.Span(arg, style={"text-align": "center"}),
            className="col-2 justify-content-center align-self-center py-1"
        ),
        html.Div(
            dcc.Dropdown(
                dd_list, default,
                className="justify-content-center",
                id=input_id,
                clearable=False,
                persistence=True,
                persistence_type="session"

            ),
            className="col-8 col-md-4 justify-content-center text-align-center py-1"
        )
    ]
    return div


def _distribution_settings():
    data = html.Div(
        [

            *_args_and_name("download-width", "Width [px]", "number", 800),
            *_args_and_name("download-height", "Height [px]", "number", 500),
            *_args_and_name("download-marker-size", "Marker Size", "number", 8),
            *_args_and_name("download-line-width", "Line Width", "number", 3),
            *_args_and_name("download-grid-width", "Grid Width", "number", 1),
            *_args_and_name("zeroline-x-width", "Zeroline X", "number", 1),
            *_args_and_name("zeroline-y-width", "Zeroline Y", "number", 1),
            *_args_and_name("d-x-tick", "X Axis dtick", "number", 1),
            *_arg_and_dropdown("Template", list(pio.templates), "plotly_white", "template-dd"),
            *_arg_and_dropdown("Name Col", [], None, "displayed-column-dd"),

            * _args_and_name("legend-vspace", "Legend Space", "number", 0.1),

        ],
        className="row p-1",
        id="distribution-settings"
    )
    return data



def figure_factory_layout():
    layout = html.Div(
        [
            dcc.Store("current-image"),
            _color_theme_modal(2),
            _modal_color_selection("primary-2"),
            _modal_color_selection("secondary-2"),
            html.Div(
                html.Div(
                    [
                        html.Div(
                            html.Div(
                                html.H4("Figure Settings"),
                                className="col-12"),
                            className="row justify-content-center"
                        ),
                        html.Div(
                            html.Div(
                                [
                                    dbc.RadioItems(
                                        options=[
                                            {'label': 'Distribution', 'value': 0},
                                            {'label': 'Heatmap', 'value': 1},
                                            {'label': 'Westernblot', 'value': 2},
                                            {'label': 'Foooooooooo', 'value': 3},
                                        ],
                                        value=0,
                                        className="d-flex justify-content-between radio-items row",
                                        labelCheckedClassName="checked-radio-text",
                                        inputCheckedClassName="checked-radio-item",
                                        id="plot-type-radio-ff",
                                        persistence_type="session",
                                        persistence=True
                                    ),
                                ],
                                className="col-10 my-2"
                            ),
                            className="row justify-content-center"
                        ),
                        html.Div(
                            [
                                html.Div(
                                    dcc.Dropdown(
                                        [],
                                        className="justify-content-center",
                                        id="protein-selector-ff",
                                        clearable=False,
                                        multi=True

                                    ),
                                    className="col-10"
                                )
                            ],

                            className="row justify-content-center p-1"
                        ),
                        html.Div(
                            html.Div(
                                [
                                    _distribution_settings(),

                                ], className="col-12 col-md-10"
                            ),

                            className="row justify-content-center"
                        ),
                        html.Div(
                            html.Div(
                                [
                                    dbc.RadioItems(
                                        options=[
                                            {'label': 'SVG', 'value': "svg"},
                                            {'label': 'PNG', 'value': "png"},
                                        ],
                                        value="svg",
                                        inline=True,
                                        className="d-flex justify-content-between radio-items",
                                        labelCheckedClassName="checked-radio-text",
                                        inputCheckedClassName="checked-radio-item",
                                        id="filetype-selector-ff",
                                        persistence_type="session",
                                        persistence=True
                                    ),
                                ],
                                className="col-10 my-2"
                            ),
                            className="row justify-content-center"
                        ),
                        html.Div(
                            [
                                html.Div(
                                    html.Button(
                                        "Select Color Scheme",
                                        style={
                                            "text-align": "center",
                                            "border": "0px solid transparent",
                                            "background": "transparent",
                                            "color": "var(--r-text-color)",
                                        },
                                        id="color-scheme2"
                                    ),
                                    className="col-4 col-md-4 justify-content-center align-self-center"
                                ),
                                html.Div(
                                    html.Button(
                                        '', id='primary-2-open-color-modal', n_clicks=0, className="btn primary-color-btn",
                                        style={"width": "100%", "height": "40px"}
                                    ),
                                    className="col-3 justify-content-center text-align-center primary-color-div primary-open-color-btn"
                                ),
                                html.Div(
                                    html.Button(
                                        '', id='secondary-2-open-color-modal', n_clicks=0,
                                        className="btn secondary-color-btn",
                                        style={"width": "100%", "height": "40px"}
                                    ),
                                    className="col-3 justify-content-center text-align-center primary-color-div secondary-open-color-btn"
                                ),

                            ],

                            className="row justify-content-center p-2"
                        ),
                    ],
                    className="databox p-2"
                ),
                className="col-12 col-lg-6"

            ),
            html.Div(
                html.Div(
                    [
                        html.Div(
                            html.H4("Preview", className="col-10 col-md-6 mt-1"),
                            className="row justify-content-center"
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [],
                                    className="col-12 col-md-11 m-2 py-2", id="figure-factory-download-preview",
                                    style={
                                        "overflow-x": "auto",
                                        "background-color": "white",
                                        "border-radius": "10px",
                                        "box-shadow": "inset black 0px 0px 10px",
                                        "border": "1px solid black"

                                    }
                                ),
                            ],
                            className="row justify-content-around",
                        ),
                    ],
                    className="databox"
                ),
                className="col-12 col-lg-6"
            )
        ],

        className="row"
    )
    return layout

layout = figure_factory_layout()


@callback(
    Output("protein-selector-ff", "options"),
    Output("protein-selector-ff", "value"),
    Input("data-store", "data"),
    State("current-protein-id", "data"),
    State("ff-ids", "data"),

)
def update_selected_proteins(rdpmsdata: RDPMSpecData, current_protein_id, selected_values):
    if rdpmsdata is None:
        raise PreventUpdate
    else:
        if selected_values is None:
            value = [rdpmsdata.df.loc[current_protein_id, "RDPMSpecID"]]
        else:
            value = selected_values
        return list(rdpmsdata.df["RDPMSpecID"]), value


@callback(
    Output("ff-ids", "data", allow_duplicate=True),
    Input("protein-selector-ff", "value"),
)
def update_ff_ids(values):
    return values

@callback(
    Output("displayed-column-dd", "options"),
    Input("data-store", "data"),

)
def update_selectable_columns(rdpmsdata):
    return list(set(rdpmsdata.extra_df) - set(rdpmsdata.score_columns))

@callback(
    Output("current-image", "data"),
    Input("protein-selector-ff", "value"),
    Input("primary-color", "data"),
    Input("secondary-color", "data"),
    Input("plot-type-radio-ff", "value"),
    Input("legend-vspace", "value"),
    Input("displayed-column-dd", "value"),
    State("data-store", "data"),
    State("unique-id", "data"),
)
def update_download_state(keys, primary_color, secondary_color, plot_type, vspace, displayed_col, rdpmsdata, uid):
    logger.info(f"selected keys: {keys}")
    if not keys:
        raise PreventUpdate
    proteins = rdpmsdata.df[rdpmsdata.df.loc[:, "RDPMSpecID"].isin(keys)].index
    logger.info(f"selected proteins: {proteins}")

    colors = primary_color, secondary_color

    fig = plot_protein_distributions(keys, rdpmsdata, colors=colors, vspace=vspace, title_col=displayed_col)
    encoded_image = Serverside(fig, key=uid + "_figure_factory")
    return encoded_image



@callback(
    Output("figure-factory-download-preview", "children"),
    Input("current-image", "data"),
    Input("filetype-selector-ff", "value"),
    Input("download-width", "value"),
    Input("download-height", "value"),
    Input("download-marker-size", "value"),
    Input("download-line-width", "value"),
    Input("download-grid-width", "value"),
    Input("zeroline-x-width", "value"),
    Input("zeroline-y-width", "value"),
    Input("d-x-tick", "value"),
    Input("template-dd", "value"),

)
def update_ff_download_preview(
        currnet_image,
        filetype,
        img_width,
        img_height,
        marker_size,
        line_width,
        grid_width,
        zeroline_x,
        zeroline_y,
        d_x_tick,
        template
):
    try:
        img_width = max(min(img_width, 2000), 100)
        img_height = max(min(img_height, 2000), 100)
    except TypeError:
        img_height = 500
        img_width = 800
    grid_width = max(0, grid_width)
    if currnet_image is None:
        raise PreventUpdate
    logger.info(f"Rendering file with width: {img_width} and height {img_height}")
    fig = currnet_image
    if template:
        fig.update_layout(template=template)
    fig.update_xaxes(dtick=d_x_tick)
    fig.update_layout(
        yaxis=dict(gridwidth=grid_width, showgrid=True if grid_width else False),
        xaxis=dict(gridwidth=grid_width, showgrid=True if grid_width else False),

    )
    fig.update_layout(
        xaxis=dict(zeroline=True if zeroline_x > 0 else False, zerolinewidth=zeroline_x,),
        yaxis=dict(zeroline=True if zeroline_y > 0 else False, zerolinewidth=zeroline_y,),
    )
    fig.update_traces(
        marker=dict(size=marker_size,
                    ),
        line=dict(
            width=line_width
        )
    )
    encoded_image = base64.b64encode(fig.to_image(format=filetype, width=img_width, height=img_height)).decode()
    fig = html.Img(
        src=f'{FILEEXT[filetype]},{encoded_image}',
        style={"margin-left": "auto", "margin-right": "auto", "display": "block"}
    )
    return fig

@callback(
    Output("color-scheme-modal-2", "is_open", allow_duplicate=True),
    Output("primary-color", "data", allow_duplicate=True),
    Output("secondary-color", "data", allow_duplicate=True),
    Input("color-scheme2", "n_clicks"),
    Input("apply-color-scheme-2", "n_clicks"),
    State("color-scheme-modal-2", "is_open"),
    State("color-scheme-dropdown-2", "value"),
    prevent_initital_call=True
)
def _open_color_theme_modal(n1, n2, is_open, selected_scheme):
    logger.info(f"{ctx.triggered_id} - triggerere color schema modal: {n1}, {n2}, {is_open}")
    if n1 == 0 or n1 is None:
        raise PreventUpdate
    if ctx.triggered_id == "color-scheme2":
        return not is_open, dash.no_update, dash.no_update
    elif ctx.triggered_id == "apply-color-scheme-2":
        if selected_scheme is None:
            raise PreventUpdate
        primary, secondary = COLOR_SCHEMES[selected_scheme]
        return not is_open, primary, secondary



@callback(
    [
        Output("secondary-2-color-modal", "is_open"),
        Output("secondary-color", "data", allow_duplicate=True),

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
        return not is_open, dash.no_update
    elif tid == "secondary-2-apply-color-modal":
        rgb = color_value["rgb"]
        r, g, b = rgb["r"], rgb["g"], rgb["b"]
        color = f"rgb({r}, {g}, {b})"
    else:
        raise ValueError("")
    return not is_open, color

@callback(
    [
        Output("primary-2-color-modal", "is_open"),
        Output("primary-color", "data", allow_duplicate=True),

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
def _toggle_secondary_color_modal(n1, n2, is_open, color_value, style):
    logger.info(f"{ctx.triggered_id} - triggered secondary color modal")
    tid = ctx.triggered_id
    if n1 == 0:
        raise PreventUpdate
    if tid == "primary-2-open-color-modal":
        return not is_open, dash.no_update
    elif tid == "primary-2-apply-color-modal":
        rgb = color_value["rgb"]
        r, g, b = rgb["r"], rgb["g"], rgb["b"]
        color = f"rgb({r}, {g}, {b})"
    else:
        raise ValueError("")
    return not is_open, color