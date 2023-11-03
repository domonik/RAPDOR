
import dash
from dash import html
import base64
from dash import dcc
from dash_extensions.enrich import callback, Input, Output, Serverside, State
from RDPMSpecIdentifier.datastructures import RDPMSpecData
import logging
import dash_bootstrap_components as dbc
import pandas as pd
from io import StringIO
from dash.exceptions import PreventUpdate
import os
from RDPMSpecIdentifier.visualize import DISABLED
from tempfile import NamedTemporaryFile
from RDPMSpecIdentifier.plots import plot_distribution
import numpy as np
from RDPMSpecIdentifier.visualize.callbacks.modalCallbacks import FILEEXT
from RDPMSpecIdentifier.visualize.modals import _get_download_input
from io import BytesIO
import plotly.io as pio
dash.register_page(__name__, path='/figure_factory')

logger = logging.getLogger(__name__)


def _args_and_name(input_id, arg, d_type, default):
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
            *_args_and_name("download-marker-size", "Marker Size", "number", 10),
            *_args_and_name("download-line-width", "Line Width", "number", 3),
            *_args_and_name("download-grid-width", "Grid Width", "number", 1),
            *_args_and_name("zeroline-x-width", "Zeroline X", "number", 1),
            *_args_and_name("zeroline-y-width", "Zeroline Y", "number", 1),
            *_arg_and_dropdown("Template", list(pio.templates), "plotly_white", "template-dd")

        ],
        className="row p-1",
        id="distribution-settings"
    )
    return data



def figure_factory_layout():
    layout = html.Div(
        [
            dcc.Store("current-image"),
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
                                    ),
                                ],
                                className="col-10 my-2"
                            ),
                            className="row justify-content-center"
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
                                    className="col-12 col-md-11 m-2", id="figure-factory-download-preview",
                                    style={
                                        "overflow-x": "auto",
                                        "background-color": "white",
                                        "border-radius": "10px"
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
    Output("current-image", "data"),
    Input("protein-selector-ff", "value"),
    Input("filetype-selector-ff", "value"),
    State("data-store", "data"),
    State("primary-color", "data"),
    State("secondary-color", "data"),
    State("unique-id", "data"),
)
def update_download_state(keys, filetype, rdpmsdata, primary_color, secondary_color, uid):
    logger.info(f"selected keys: {keys}")
    if not keys:
        raise PreventUpdate
    proteins = rdpmsdata.df[rdpmsdata.df.loc[:, "RDPMSpecID"].isin(keys)].index
    logger.info(f"selected proteins: {proteins}")

    colors = primary_color, secondary_color
    array, _ = rdpmsdata[proteins[0]]
    if rdpmsdata.state.kernel_size is not None:
        i = int(rdpmsdata.state.kernel_size // 2)
    else:
        i = 0
    fig = plot_distribution(array, rdpmsdata.internal_design_matrix, groups="RNase", offset=i, colors=colors)
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

