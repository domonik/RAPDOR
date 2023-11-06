
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
from RDPMSpecIdentifier.plots import plot_protein_distributions, plot_protein_westernblots
import numpy as np
from RDPMSpecIdentifier.visualize.callbacks.modalCallbacks import FILEEXT
from RDPMSpecIdentifier.visualize.modals import _color_theme_modal, _modal_color_selection
from io import BytesIO
import plotly.io as pio
import copy

dash.register_page(__name__, path='/figure_factory')

logger = logging.getLogger(__name__)

pio.templates["FFDefault"] = copy.deepcopy(pio.templates["plotly_white"])

pio.templates["FFDefault"].update(
    {
        "layout": {
            # e.g. you want to change the background to transparent
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": " rgba(0,0,0,0)",
            "font": dict(color="black"),
            "xaxis": dict(linecolor="black", showline=True),
            "yaxis": dict(linecolor="black", showline=True)
        }
    }
)



def _arg_x_and_y(input_id_x, input_id_y, arg, d_type, default_x, default_y):
    if isinstance(default_x, int):
        step = 1
    elif isinstance(default_x, float):
        step = 0.01
    else:
        step = None
    div = [
        html.Div(
            html.Span(arg, style={"text-align": "center"}),
            className="col-4 col-md-2 justify-content-center align-self-center py-1"
        ),
        html.Div(
            html.Div(
            [
                html.Div(
                    html.Span("X", style={"text-align": "center"}),

                    className="col-1 p-0 align-self-center"
                ),
                html.Div(
                    dcc.Input(
                        style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white",
                               "text-align": "center"},
                        id=input_id_x,
                        className="text-align-center",
                        value=default_x,
                        type=d_type,
                        step=step,
                        persistence=True,
                        persistence_type="session"
                    ),
                    className="col-4 p-0"
                ),
                html.Div(
                    html.Span("Y", style={"text-align": "center"}),

                    className="col-1 p-0 align-self-center"
                ),
                html.Div(
                    dcc.Input(
                        style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white",
                               "text-align": "center"},
                        id=input_id_y,
                        className="text-align-center",
                        value=default_y,
                        type=d_type,
                        step=step,
                        persistence=True,
                        persistence_type="session"
                    ),
                    className="col-4 p-0"
                )

            ],
                className="row m-0 p-0 justify-content-between"
            ),
            className="col-8 col-md-4 justify-content-center text-align-center align-self-center py-1"
        ),
    ]
    return div

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
                className="col-4 col-md-2 justify-content-center align-self-center py-1"
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
                    persistence_type="session",
                ),
                className="col-8 col-md-4 justify-content-center text-align-center py-1"
            )
        ]
    return div

def _arg_and_dropdown(arg, dd_list, default, input_id):
    div = [
        html.Div(
            html.Span(arg, style={"text-align": "center"}),
            className="col-4 col-md-2 justify-content-center align-self-center py-1"
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

            *_arg_and_dropdown(
                "Template",
                ["FFDefault"] + [template for template in list(pio.templates) if template != "FFDefault"],
                "FFDefault", "template-dd"
            ),
            *_arg_and_dropdown("Name Col", ["RDPMSpecID"], "RDPMSpecID", "displayed-column-dd"),
            *_args_and_name("download-width", "Width [px]", "number", 800),
            *_args_and_name("download-height", "Height [px]", "number", 500),
            *_args_and_name("download-marker-size", "Marker Size", "number", 8),
            *_args_and_name("download-line-width", "Line Width", "number", 3),
            *_args_and_name("download-grid-width", "Grid Width", "number", 1),
            *_args_and_name("v-space", "Vertical Space", "number", 0.01),
            *_arg_x_and_y("legend1-x", "legend1-y", "Legend Pos", "number", 0., 1.),
            *_arg_x_and_y("legend2-x", "legend2-y", "Legend2 Pos", "number", 0., 1.),
            *_arg_x_and_y("d-x-tick", "d-y-tick", "Axid dtick", "number", 1, 1),
            *_arg_x_and_y("zeroline-x-width", "zeroline-y-width", "Zeroline", "number", 1, 0),

        ],
        className="row p-5 p-md-1",
        id="distribution-settings"
    )
    return data

def _font_settings():
    data = html.Div(
        [

            html.Div(html.H5("Fonts"), className="col-12 justify-content-center "),
            *_args_and_name("legend-font-size", "Legend", "number", 12),
            *_args_and_name("axis-font-size", "Axis", "number", 18),

        ],
        className="row p-5 p-md-1",
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
                                            {'label': 'Westernblot', 'value': 2},
                                        ],
                                        value=0,
                                        className="d-flex justify-content-around radio-items row",
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
                                    _font_settings()

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
    State("current-row-ids", "data"),

)
def update_selected_proteins(rdpmsdata: RDPMSpecData, current_row_ids):
    if rdpmsdata is None:
        raise PreventUpdate
    else:
        if current_row_ids is not None:
            value = list(rdpmsdata.df.loc[current_row_ids, "RDPMSpecID"])
        else:
            value = dash.no_update

        return list(rdpmsdata.df["RDPMSpecID"]), value


@callback(
    Output("current-row-ids", "data", allow_duplicate=True),
    Input("protein-selector-ff", "value"),
    State("data-store", "data"),

)
def update_row_ids(values, rdpmsdata):
    values = list(rdpmsdata.df[rdpmsdata.df["RDPMSpecID"].isin(values)].index)
    return values

@callback(
    Output("displayed-column-dd", "options"),
    Input("data-store", "data"),

)
def update_selectable_columns(rdpmsdata):
    return list(set(rdpmsdata.extra_df) - set(rdpmsdata.score_columns))

@callback(
    Output("current-image", "data"),
    Output("download-marker-size", "value"),
    Output("download-marker-size", "disabled"),
    Output("download-line-width", "value"),
    Output("download-line-width", "disabled"),
    Input("protein-selector-ff", "value"),
    Input("primary-color", "data"),
    Input("secondary-color", "data"),
    Input("plot-type-radio-ff", "value"),
    Input("displayed-column-dd", "value"),
    Input("v-space", "value"),
    State("data-store", "data"),
    State("unique-id", "data"),
)
def update_download_state(keys, primary_color, secondary_color, plot_type, displayed_col, vspace, rdpmsdata, uid):
    logger.info(f"selected keys: {keys}")
    if not keys:
        raise PreventUpdate
    proteins = rdpmsdata.df[rdpmsdata.df.loc[:, "RDPMSpecID"].isin(keys)].index
    logger.info(f"selected proteins: {proteins}")

    colors = primary_color, secondary_color
    if plot_type == 2:
        fig = plot_protein_westernblots(keys, rdpmsdata, colors=colors, title_col=displayed_col, vspace=vspace)
        settings = DEFAULT_WESTERNBLOT_SETTINGS
    else:
        fig = plot_protein_distributions(keys, rdpmsdata, colors=colors, title_col=displayed_col, vspace=vspace)
        settings = DEFAULT_DISTRIBUTION_SETTINGS
    encoded_image = Serverside(fig, key=uid + "_figure_factory")
    return encoded_image, *settings

DEFAULT_DISTRIBUTION_SETTINGS = (8, False, 3, False)
DEFAULT_WESTERNBLOT_SETTINGS = (None, True, None, True)


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
    Input("d-y-tick", "value"),
    Input("legend1-x", "value"),
    Input("legend1-y", "value"),
    Input("legend2-x", "value"),
    Input("legend2-y", "value"),
    Input("template-dd", "value"),
    Input("legend-font-size", "value"),
    Input("axis-font-size", "value"),

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
        d_y_tick,
        lx,
        ly,
        l2x,
        l2y,
        template,
        legend_font_size,
        axis_font_size
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
    fig.update_yaxes(dtick=d_y_tick)
    fig.update_xaxes(titlefont=dict(size=axis_font_size))
    fig.update_yaxes(titlefont=dict(size=axis_font_size))
    fig.update_annotations(
        font=dict(size=axis_font_size)
    )
    fig.update_xaxes(zeroline=True if zeroline_x > 0 else False, zerolinewidth=zeroline_x,)
    fig.update_yaxes(zeroline=True if zeroline_y > 0 else False, zerolinewidth=zeroline_y,)
    fig.update_yaxes(gridwidth=grid_width, showgrid=True if grid_width else False)
    fig.update_xaxes(gridwidth=grid_width, showgrid=True if grid_width else False)
    fig.update_layout(
        legend2=dict(
            y=l2y,
            x=l2x
        ),
        legend=dict(
            x=lx,
            y=ly
        )
    )
    if marker_size is not None:
        if marker_size > 0:
            fig.update_traces(
                marker=dict(size=marker_size)
            )
        else:
            fig.update_traces(mode="lines")
    if line_width is not None:
        fig.update_traces(
            line=dict(width=max(line_width, 0)
                      )
        )
    fig.update_layout(
        legend=dict(
            font=dict(size=legend_font_size)
        ),
        legend2=dict(
            font=dict(size=legend_font_size)
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