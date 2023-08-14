import time
import re
import pandas as pd
from dash import dcc, dash_table
from dash import html, ctx
import logging
from dash.dependencies import Input, Output, State
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from RDPMSpecIdentifier.plots import plot_distribution, plot_heatmap, plot_barcode_plot, plot_replicate_distribution
from RDPMSpecIdentifier.datastructures import RDPMSpecData
from dash import clientside_callback, ClientsideFunction
import os
import dash
import plotly.io as pio
import plotly.graph_objs as go
from pandas.api.types import is_numeric_dtype
import dash_loading_spinners as dls
import numpy as np
from time import sleep
import base64
import tempfile
import dash_daq as daq
import RDPMSpecIdentifier

VERSION = RDPMSpecIdentifier.__version__

FILEDIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(FILEDIR, "assets")
TMPDIR = tempfile.TemporaryDirectory(suffix="RDPMSpec")

LOGO = os.path.join(ASSETS_DIR, "RDPMSpecIdentifier_dark_no_text.svg")
LIGHT_LOGO = os.path.join(ASSETS_DIR, "RDPMSpecIdentifier_light_no_text.svg")
assert os.path.exists(LOGO), f"{LOGO} does not exist"
assert os.path.exists(LIGHT_LOGO), f"{LIGHT_LOGO} does not exist"
encoded_img = base64.b64encode(open(LOGO, 'rb').read())


img_text = open(LOGO, 'r').read()
color = "fill:#ff8add"
res = re.search(color, img_text)
COLOR_IDX = res.start()
COLOR_END =res.end()

logger = logging.getLogger("RDPMSpecIdentifier")

app = dash.Dash(
    "RDPMSpecIdentifier Dashboard",
    title="RDPMSpec Visualizer",
    external_stylesheets=[dbc.themes.DARKLY],
    #assets_url_path=ASSETS_DIR,
    assets_folder=ASSETS_DIR,
    index_string=open(os.path.join(ASSETS_DIR, "index.html")).read(),
)

pio.templates["plotly_white"].update(
    {
        "layout": {
            # e.g. you want to change the background to transparent
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": " rgba(0,0,0,0)",
            "font": dict(color="white"),
        }
    }
)

DEFAULT_COLORS = {"primary": "rgb(138, 255, 172)", "secondary": "rgb(255, 138, 221)"}


def _header_layout():
    svg = 'data:image/svg+xml;base64,{}'.format(encoded_img.decode())
    header = html.Div(
        html.Div(
            html.Div(
                [
                    html.Div(className="col-md-3 col-0"),
                    html.Div(
                        html.Img(src=svg, style={"width": "20%", "min-width": "300px"}, className="p-1"),
                        className="col-md-6 col-11 justify-content-center justify-conent-md-start", id="logo-container"
                    ),
                    html.Div(
                        daq.BooleanSwitch(
                            label='',
                            labelPosition='left',
                            color="var(--r-text-color)",
                            on=True,
                            id="night-mode",
                            className="align-self-center px-2",
                            persistence=True

                        ),
                        className="col-1 col-md-3 d-flex justify-content-end justify-self-end"
                    )


                ],
                className="row"
            ),
            className="databox header-box",
            style={"text-align": "center"},
        ),
        className="col-12 m-0 px-0 justify-content-center"
    )
    return header


def distribution_panel(data):
    sel_data = data.df.index[0:100]

    distribution_panel = html.Div(
        [
            html.Div(
                [

                    html.Div(
                        [
                            html.Div(
                                html.Div(id="placeholder2"),

                                className="col-0 col-md-4", id="placeholder"
                            ),
                            html.Div(
                                html.H4(f"Protein {sel_data[0]}", style={"text-align": "center"}, id="protein-id"),
                                className="col-12 col-md-4 justify-content-center align-self-center",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(className="col-md-1 col-0"),
                                            html.Div(
                                                html.Span("Replicate Mode", className="align-self-center"),
                                                className="col-4 col-md-4 d-flex align-items-bottom justify-content-center"
                                            ),
                                            html.Div(

                                                daq.BooleanSwitch(
                                                    label='',
                                                    labelPosition='left',
                                                    color="var(--primary-color)",
                                                    on=False,
                                                    id="replicate-mode",
                                                    className="align-self-center",

                                                ),
                                                className="col-2 col-md-2 d-flex align-items-center justify-content-center"
                                            ),
                                            html.Div(
                                                html.Button("Download Image", style={"text-align": "center"},
                                                            id="open-modal", className="btn btn-primary"),
                                                className="col-6 col-md-5 justify-content-right align-self-center text-end",
                                            ),

                                        ],
                                        className="row justify-content-right"
                                    ),

                                ],
                                className="col-12 col-md-4"
                            ),

                            dcc.Download(id="download-image"),


                        ],
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Graph(id="distribution-graph", style={"height": "320px"}),
                                className="col-12"
                            ),
                        ],
                        className="row justify-content-center"
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Graph(id="westernblot-graph", style={"height": "70px"}),
                                className="col-12"
                            ),
                            html.Div("Fraction", className="col-12 pt-2", style={"text-align": "center", "font-size": "20px"})
                        ],
                        className="row justify-content-center pb-3"
                    ),

                ],
                className="databox",
            )
        ],
        className="col-12 p-1 justify-content-center"
    )
    return distribution_panel


def selector_box(data):
    sel_box = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        html.Div(
                            html.H4("Settings", style={"text-align": "center"}),
                            className="col-12 justify-content-center"
                        ),
                        className="row justify-content-center p-2 p-md-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Span("Distance Method", style={"text-align": "center"}),
                                className="col-3 col-md-3 justify-content-center align-self-center"
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    list(rdpmsdata.methods.keys()), list(rdpmsdata.methods.keys())[0],
                                    className="justify-content-center",
                                    id="distance-method",
                                    clearable=False

                                ),
                                className="col-7 justify-content-center text-align-center"
                            )
                        ],
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Span("Kernel Size", style={"text-align": "center"}),
                                className="col-10 col-md-3 justify-content-center align-self-center"
                            ),
                            html.Div(
                                dcc.Slider(
                                    0, 5, step=None,
                                    marks={
                                        0: "0",
                                        3: '3',
                                        5: '5',
                                    }, value=3,
                                    className="justify-content-center",
                                    id="kernel-slider"
                                ),
                                className="col-10 col-md-7 justify-content-center",
                            ),
                        ],
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        html.Div(
                            html.Button('Get Score', id='score-btn', n_clicks=0, className="btn btn-primary", style={"width": "100%"}),
                            className="col-10 justify-content-center text-align-center"
                        ),
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        html.Div(
                            html.Button('Rank Table', id='rank-btn', n_clicks=0, className="btn btn-primary",
                                        style={"width": "100%"}),
                            className="col-10 justify-content-center text-align-center"
                        ),
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Input(
                                    style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white",
                                           "text-align": "center"},
                                    id="distance-cutoff",
                                    placeholder="Distance Cutoff",
                                    className="text-align-center",
                                    type="number",
                                    min=0,
                                ),
                                className="col-3 text-align-center align-items-center"
                            ),
                            html.Div(
                                html.Button('Peak T-Tests', id='local-t-test-btn', n_clicks=0,
                                            className="btn btn-primary",
                                            style={"width": "100%"}),
                                className="col-7 justify-content-center text-align-center"
                            ),
                        ],
                        className="row justify-content-center p-2"
                    ),

                    html.Div(
                        [
                            html.Div(
                                dcc.Input(
                                    style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white", "text-align": "center"},
                                    id="permanova-permutation-nr",
                                    placeholder="Number of Permutations",
                                    className="text-align-center",
                                    type="number",
                                    min=1
                                ),
                                className="col-3 text-align-center align-items-center"
                            ),
                            html.Div(
                                html.Button('Run PERMANOVA', id='permanova-btn', n_clicks=0,
                                            className="btn btn-primary",
                                            style={"width": "100%"}),
                                className="col-7 justify-content-center text-align-center"
                            ),
                            html.Div(
                                id="alert-div",
                                className="col-10"
                            )

                        ],
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Input(
                                    style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white",
                                           "text-align": "center"},
                                    id="anosim-permutation-nr",
                                    placeholder="Number of Permutations",
                                    className="text-align-center",
                                    type="number",
                                    min=1
                                ),
                                className="col-3 text-align-center align-items-center"
                            ),
                            html.Div(
                                html.Button('Run ANOSIM', id='anosim-btn', n_clicks=0,
                                            className="btn btn-primary",
                                            style={"width": "100%"}),
                                className="col-7 justify-content-center text-align-center"
                            ),

                        ],
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Button('Export TSV', id='export-btn', n_clicks=0, className="btn btn-primary",
                                            style={"width": "100%"}),
                                className="col-10 justify-content-center text-align-center"
                            ),
                            dcc.Download(id="download-dataframe-csv"),
                        ],

                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Span("Select Color Scheme", style={"text-align": "center"}, id="color-scheme"),
                                className="col-4 col-md-4 justify-content-center align-self-center"
                            ),
                            html.Div(
                                html.Button(
                                    '', id='primary-open-color-modal', n_clicks=0, className="btn btn-primary",
                                    style={"width": "100%", "height": "40px", "background-color": DEFAULT_COLORS["primary"]}
                                ),
                                className="col-3 justify-content-center text-align-center primary-color-div"
                            ),
                            html.Div(
                                html.Button(
                                    '', id='secondary-open-color-modal', n_clicks=0,
                                    className="btn btn-primary",
                                    style={"width": "100%", "height": "40px", "background-color": DEFAULT_COLORS["secondary"]}
                                ),
                                className="col-3 justify-content-center text-align-center primary-color-div"
                            ),

                        ],

                        className="row justify-content-center p-2"
                    ),
                ],
                className="databox justify-content-center"
            )
        ],
        className="col-12 col-md-6 p-1 justify-content-center equal-height-column"
    )
    return sel_box


def _get_table(rbmsdata: RDPMSpecData):
    table = html.Div(
        [
            html.Div(
                html.Div(
                        [

                            dls.RingChase(
                                html.Div(
                                    _create_table(rbmsdata),
                                    className="col-12 justify-content-center",
                                    id="data-table"

                                ),
                                color="var(--primary-color)",
                                width=200,
                                thickness=20,

                            ),
                            html.Div(
                                dcc.Dropdown(
                                    rdpmsdata.extra_df.columns,
                                    placeholder="Select Table Columns",
                                    className="justify-content-center",
                                    multi=True,
                                    id="table-selector"
                                ),
                                className="col-12 pt-1"
                            ),

                        ],


                    className="row justify-content-center"
                ),

                className="databox p-3",
            )
        ],
        className="col-12 p-1 justify-content-center",
    )
    return table


def _create_table(rbmsdata, selected_columns = None):
    global data
    if selected_columns is None:
        selected_columns = []

    data = rdpmsdata.extra_df.loc[:, rdpmsdata.id_columns + selected_columns]
    for name in rdpmsdata.calculated_score_names:
        if name in rdpmsdata.extra_df:
            data = pd.concat((data, rdpmsdata.extra_df[name]), axis=1)
    columns = []
    num_cols = ["shift direction"]
    for i in data.columns:
        if i != "id":
            d = dict()
            d["name"] = str(i)
            d["id"] = str(i)
            if is_numeric_dtype(data[i]):
                d["type"] = "numeric"
                if "p-Value" in i:
                    d["format"] = Format(precision=2)
                else:
                    d["format"] = Format(precision=4)

                num_cols.append(str(i))
            columns.append(d)
    t = dash_table.DataTable(
        data.to_dict('records'),
        columns,
        id='tbl',
        sort_action="custom",
        sort_mode="multi",
        sort_by=[],

        filter_action='custom',
        filter_query='',
        page_size=50,
        page_current=0,
        page_action="custom",
        style_table={'overflowX': 'auto', "padding": "1px", "height": "300px",
                     "overflowY": "auto"},
        fixed_rows={'headers': True},
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white',
            "border": "1px",
            "font-family": "var(--bs-body-font-family)"

        },
        style_data={
            'color': 'var(--r-text-color)',
            "border": "1px",
            "font-family": "var(--bs-body-font-family)"

        },
        style_data_conditional=SELECTED_STYLE,
        style_cell={
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'maxWidth': 0
        },
        style_filter={
            "color": "white",
            "border-color": "red"
        },
        style_cell_conditional=[
                                   {
                                       'if': {'column_id': 'RDPMSpecID'},
                                       'textAlign': 'left',
                                       "width": "10%"
                                   }
                               ]
    ),

    return t


def correlation_heatmap_box():
    heatmap_box = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        dls.RingChase(
                            [
                                html.Div(
                                    html.H4(
                                        "Distance",
                                        id="distance-header"
                                    ),
                                    className="col-12 pb-2"
                                ),
                                html.Div(
                                    dcc.Graph(id="heatmap-graph", style={"height": "370px"}),
                                    className="col-12"
                                ),

                            ],
                            color="var(--primary-color)",
                            width=200,
                            thickness=20,
                        ),

                       className="row p-2 justify-content-center",
                    ),

                ],
                className="databox",
            )
        ],
        className="col-12 col-md-6 p-1 justify-content-center equal-height-column"
    )
    return heatmap_box


def _get_app_layout(dash_app):
    dash_app.layout = html.Div(
        [
            html.Div(id="recomputation"),
            html.Div(
                _header_layout(),
                className="row px-0 justify-content-center align-items-center sticky-top"
            ),
            html.Div(
                distribution_panel(rdpmsdata),
                className="row px-2 justify-content-center align-items-center"

            ),
            html.Div(
                _get_table(rdpmsdata),
                className="row px-2 justify-content-center align-items-center",
                id="protein-table"
            ),
            html.Div(
                [correlation_heatmap_box(), selector_box(rdpmsdata)],
                className="row px-2 row-eq-height justify-content-center"
            ),
            html.Div(
                _footer(),
                className="row px-3 py-3 mt-2 justify-content-end align-items-center",
                style={
                    "background-color": "var(--databox-color)",
                    "border-color": "black",
                    "border-width": "2px",
                    "border-style": "solid",
                },
            ),
            _modal_image_download(),
            _modal_color_selection("primary"),
            _modal_color_selection("secondary")

        ],
        className="container-fluid"
    )


def _footer():
    footer = [
        html.Div(
            [
                html.P(f"Version {VERSION}", className="text-end"),
                html.P(
                    html.A(
                        f"GitHub",
                        className="text-end",
                        href="https://github.com/domonik/RDPMSpecIdentifier",
                        target="_blank"
                    ),
                    className="text-end")
            ],
            className="col-12 col-md-4 flex-column justify-content-end align-items-end"
        )
    ]
    return footer

def _modal_image_download():
    modal = dbc.Modal(
        [
            dbc.ModalHeader("Select file Name"),
            dbc.ModalBody(
                [
                    html.Div(
                        [
                            html.Div(dbc.Input("named-download",),
                                        className=" col-9"),
                            dbc.Button("Download", id="download-image-button", className="btn btn-primary col-3"),
                        ],
                        className="row justify-content-around",
                    )
                ]
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="close", className="ml-auto",
                           n_clicks=0)),
        ],
        id="modal",
    )
    return modal


def _modal_color_selection(number):
    color = DEFAULT_COLORS[number]
    color = color.split("(")[-1].split(")")[0]
    r, g, b = (int(v) for v in color.split(","))
    modal = dbc.Modal(
        [
            dbc.ModalHeader("Select color"),
            dbc.ModalBody(
                [
                    html.Div(
                        [
                            daq.ColorPicker(
                                id=f'{number}-color-picker',
                                label='Color Picker',
                                size=400,
                                theme={"dark": True},
                                value={"rgb": dict(r=r, g=g, b=b, a=1)}
                            ),
                        ],
                        className="row justify-content-around",
                    )
                ]
            ),
            dbc.ModalFooter(
                dbc.Button("Apply", id=f"{number}-apply-color-modal", className="ml-auto",
                           n_clicks=0)),
        ],
        id=f"{number}-color-modal",
    )
    return modal

@app.callback(
    Output("recomputation", "children"),
    Input("kernel-slider", "value"),
    Input("distance-method", "value")
)
def recompute_data(kernel_size, distance_method):
    if kernel_size == 0:
        kernel_size = None
    method = rdpmsdata.methods[distance_method]
    eps = 0 if distance_method == "Jensen-Shannon-Distance" else 10 # Todo: Make this optional
    rdpmsdata.normalize_and_get_distances(method=method, kernel=kernel_size, eps=eps)
    return html.Div()


@app.callback(
    Output("logo-container", "children"),
    Input("night-mode", "on"),
    Input("secondary-open-color-modal", "style"),
)
def update_logo(night_mode, style):
    color2 = style["background-color"]
    rep = f"fill:{color2}"
    l_image_text = img_text[:COLOR_IDX] + rep + img_text[COLOR_END:]
    if not night_mode:
        l_image_text = re.sub("fill:#f2f2f2", "fill:black", l_image_text)
    encoded_img = base64.b64encode(l_image_text.encode())
    img = 'data:image/svg+xml;base64,{}'.format(encoded_img.decode())
    return html.Img(src=img, style={"width": "20%", "min-width": "300px"}, className="p-1"),



@app.callback(
    Output("distribution-graph", "figure"),
    [
        Input("protein-id", "children"),
        Input('recomputation', 'children'),
        Input("primary-open-color-modal", "style"),
        Input("secondary-open-color-modal", "style"),
        Input("replicate-mode", "on"),
        Input("night-mode", "on")
    ],

)
def update_plot(key, kernel_size, primary_color, secondary_color, replicate_mode, night_mode):
    colors = primary_color['background-color'], secondary_color['background-color']
    key = key.split("Protein ")[-1]
    if key is None:
        raise PreventUpdate
    array, _ = rdpmsdata[key]
    i = 0
    if rdpmsdata.current_kernel_size is not None:
        i = int(np.floor(rdpmsdata.current_kernel_size / 2))
    if replicate_mode:
        fig = plot_replicate_distribution(array, rdpmsdata.internal_design_matrix, groups="RNAse", offset=i, colors=colors)
    else:
        fig = plot_distribution(array, rdpmsdata.internal_design_matrix, groups="RNAse", offset=i, colors=colors)
    fig.layout.template = "plotly_white"
    if not night_mode:
        fig.update_layout(
            font=dict(color="black"),
            yaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),
            xaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),

        )
    fig.update_layout(
        margin={"t": 0, "b": 30, "r": 50},
        font=dict(
            size=16,
        )
    )
    fig.update_xaxes(dtick=1)
    fig.update_xaxes(fixedrange=True)
    return fig

@app.callback(
    Output("westernblot-graph", "figure"),
    [
        Input("protein-id", "children"),
        Input('recomputation', 'children'),
        Input("primary-open-color-modal", "style"),
        Input("secondary-open-color-modal", "style"),
        Input("night-mode", "on"),
    ],

)
def update_westernblot(key, kernel_size, primary_color, secondary_color, night_mode):
    colors = primary_color['background-color'], secondary_color['background-color']
    key = key.split("Protein ")[-1]
    if key is None:
        raise PreventUpdate
    array = rdpmsdata.array[rdpmsdata.df.index.get_loc(key)]

    fig = plot_barcode_plot(array, rdpmsdata.internal_design_matrix, groups="RNAse", colors=colors)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_xaxes(showgrid=False, showticklabels=False)
    if not night_mode:
        fig.update_layout(
            font=dict(color="black"),
            yaxis=dict(gridcolor="black"),
            xaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),

        )
    fig.update_layout(
        margin={"t": 0, "b": 0, "r": 50},
        font=dict(
            size=16,
        )
    )
    fig.update_xaxes(fixedrange=True)

    fig.layout.template = "plotly_white"
    return fig


@app.callback(
    [
        Output("heatmap-graph", "figure"),
        Output("distance-header", "children")
    ],
    [
        Input("protein-id", "children"),
        Input('recomputation', 'children'),
        Input("primary-open-color-modal", "style"),
        Input("secondary-open-color-modal", "style"),
        Input("night-mode", "on"),

    ],
    State("distance-method", "value")

)
def update_heatmap(key, kernel_size, primary_color, secondary_color, night_mode, distance_method):
    colors = primary_color['background-color'], secondary_color['background-color']
    key = key.split("Protein ")[-1]
    if key is None:
        raise PreventUpdate
    _, distances = rdpmsdata[key]
    fig = plot_heatmap(distances, rdpmsdata.internal_design_matrix, groups="RNAse", colors=colors)
    fig.layout.template = "plotly_white"
    if not night_mode:
        fig.update_layout(
            font=dict(color="black"),
            yaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),
            xaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),

        )
    fig.update_layout(
        margin={"t": 0, "b": 0, "l": 0, "r": 0}
    )
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    return fig, f"Sample {distance_method}"





SELECTED_STYLE = [
        {
            "if": {"state": "active"},
            "backgroundColor": "rgba(150, 180, 225, 0.2)",
            "border-top": "2px solid var(--primary-color)",
            "border-bottom": "2px solid var(--primary-color)",
            "border-left": "0px solid var(--primary-color)",
            "border-right": "0px solid var(--primary-color)",
        },
        {
            "if": {"state": "selected"},
            "backgroundColor": "rgba(14, 102, 232, 1) !important",
            "border-top": "2px solid var(--primary-color)",
            "border-bottom": "2px solid var(--primary-color)",
            "border-left": "0px solid var(--primary-color)",
            "border-right": "0px solid var(--primary-color)",
        },
    ]

@app.callback(
    Output("tbl", "style_data_conditional"),
    Input('tbl', 'active_cell'),
    Input('tbl', 'data'),
    State("protein-id", "children"),
    State("tbl", "page_size"),
    State('tbl', "page_current"),

)
def style_selected_col(active_cell, sort_by, key, page_size, current_page):

    if "tbl.data" in ctx.triggered_prop_ids:
        key = key.split("Protein ")[-1]
        if key in data.index:
            loc = data.index.get_loc(key)
            page = int(np.floor(loc / page_size))
            if page != current_page:
                row_idx = -1
            else:
                row_idx = int(loc % page_size)
        else:
            row_idx = -1
    else:
        if active_cell is None:
            raise PreventUpdate
        row_idx = active_cell["row"]

    style = [
        {
            "if": {"row_index": row_idx},
            "backgroundColor": "red !important",
            "border-top": "2px solid var(--primary-color)",
            "border-bottom": "2px solid var(--primary-color)",
            "border-left": "0px solid var(--primary-color)",
            "border-right": "0px solid var(--primary-color)",
        },
    ]
    style_data_conditional = SELECTED_STYLE + style
    return style_data_conditional

@app.callback(
        Output("protein-id", "children"),

    [
        Input('tbl', 'active_cell'),
    ],

)
def update_selected_id(active_cell):

    if active_cell is None:
        raise PreventUpdate
    active_row_id = active_cell["row_id"]
    active_row_id = f"Protein {active_row_id}"


    return active_row_id


@app.callback(
    [
        Output("data-table", "children"),
        Output("alert-div", "children"),
        Output('tbl', 'sort_by'),
    ],
    [
        Input('table-selector', 'value'),
        Input('score-btn', 'n_clicks'),
        Input('permanova-btn', 'n_clicks'),
        Input('anosim-btn', 'n_clicks'),
        Input('local-t-test-btn', 'n_clicks'),
        Input("recomputation", "children"),
        Input("rank-btn",  "n_clicks")

    ],
    [
        State("permanova-permutation-nr", "value"),
        State("anosim-permutation-nr", "value"),
        State("distance-cutoff", "value"),
        State('tbl', 'sort_by'),

    ]

)
def new_columns(sel_columns, n_clicks, permanova_clicks, anosim_clicks, t_test_clicks, recompute, ranking, permanova_permutations, anosim_permutations, distance_cutoff, current_sorting):
    alert = False
    if ctx.triggered_id == "rank-btn":
        try:
            cols = [col['column_id'] for col in current_sorting if col != "Rank"]
            asc = [col['direction'] == "asc" for col in current_sorting if col != "Rank"]

            rdpmsdata.rank_table(cols, asc)
        except Exception as e:
            alert = True
            alert_msg = f"Ranking Failed:\n{str(e)}"

    if ctx.triggered_id == "permanova-btn":

        if permanova_clicks == 0:
            raise PreventUpdate
        else:
            if permanova_permutations is None:
                permanova_permutations = 9999
            if rdpmsdata.permutation_sufficient_samples:
                rdpmsdata.calc_permanova_p_value(permutations=permanova_permutations, threads=os.cpu_count(), mode="local")
            else:
                rdpmsdata.calc_permanova_p_value(permutations=permanova_permutations, threads=os.cpu_count(), mode="global")

                alert = True
                alert_msg = "Insufficient Number of Samples per Groups. P-Value is derived using all Proteins as background."
                " This might be unreliable"
    if ctx.triggered_id == "anosim-btn":
        if anosim_clicks == 0:
            raise PreventUpdate
        else:
            if anosim_permutations is None:
                anosim_permutations = 9999
            if rdpmsdata.permutation_sufficient_samples:
                rdpmsdata.calc_anosim_p_value(permutations=anosim_permutations, threads=os.cpu_count(), mode="local")
            else:
                rdpmsdata.calc_anosim_p_value(permutations=anosim_permutations, threads=os.cpu_count(), mode="global")
                alert = True
                alert_msg = "Insufficient Number of Samples per Groups. P-Value is derived using all Proteins as background."
                " This might be unreliable"
    if ctx.triggered_id == "local-t-test-btn":
        if "RNAse True peak pos" not in rdpmsdata.df:
            rdpmsdata.determine_peaks()
        rdpmsdata.calc_welchs_t_test(distance_cutoff=distance_cutoff)

    if ctx.triggered_id == "score-btn":
        if n_clicks == 0:
            raise PreventUpdate
        else:
            rdpmsdata.calc_all_scores()
    if alert:
        alert_msg = html.Div(
            dbc.Alert(
                alert_msg,
                color="danger",
                dismissable=True,
            ),
            className="p-2 align-items-center, alert-msg",

        )
    else:
        alert_msg = []

    return _create_table(rdpmsdata, sel_columns), alert_msg, current_sorting


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export-btn", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(rdpmsdata.extra_df.to_csv, "RDPMSpecIdentifier.tsv", sep="\t")

@app.callback(
    Output('tbl', 'data'),
    Output('tbl', "page_current"),

    Input('tbl', "page_current"),
    Input('tbl', "page_size"),
    Input('tbl', 'sort_by'),
    Input('tbl', 'filter_query'),
    State('table-selector', 'value'),
    State("protein-id", "children"),

)
def update_table(page_current, page_size, sort_by, filter, selected_columns, key):
    key = key.split("Protein ")[-1]
    global data
    if selected_columns is None:
        selected_columns = []

    data = rdpmsdata.extra_df.loc[:, rdpmsdata.id_columns + selected_columns]
    for name in rdpmsdata.calculated_score_names:
        if name in rdpmsdata.extra_df:
            data = pd.concat((data, rdpmsdata.extra_df[name]), axis=1)

    filtering_expressions = filter.split(' && ')
    logger.debug(f"filter expressions: {filtering_expressions}")
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            data = data.loc[getattr(data[col_name], operator)(filter_value)]
        elif operator == 'contains':
            filter_value = str(filter_value).split(".0")[0]
            data = data.loc[data[col_name].str.contains(filter_value).fillna(False)]
        elif operator == 'datestartswith':
            filter_value = str(filter_value).split(".0")[0]

            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            data = data.loc[data[col_name].str.startswith(filter_value)]

    if sort_by is not None:
        if len(sort_by):
            data = data.sort_values(
                [col['column_id'] for col in sort_by],
                ascending=[
                    col['direction'] == 'asc'
                    for col in sort_by
                ],
                inplace=False
            )
    if "tbl.page_current" in ctx.triggered_prop_ids or "tbl.sort_by" in ctx.triggered_prop_ids:
        page = page_current
        size = page_size
    elif key in data.index:
        loc = data.index.get_loc(key)
        page = int(np.floor(loc / page_size))
        size = page_size
    else:
        page = page_current
        size = page_size



    return data.iloc[page * size: (page + 1) * size].to_dict('records'), page


operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]

def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3

@app.callback(
    [
        Output("modal", "is_open"),
        Output("named-download", "value")
     ],
    [
        Input("open-modal", "n_clicks"),
        Input("close", "n_clicks"),
        Input("download-image-button", "n_clicks"),
    ],
    [State("modal", "is_open"),
     State("protein-id", "children")
     ],
    prevent_initial_call=True

)
def _toggle_modal(n1, n2, n3, is_open, key):
    key = key.split("Protein ")[-1]
    filename = key + ".svg"
    if n1 or n2 or n3:
        return not is_open, filename
    return is_open, filename


@app.callback(
    [
        Output("primary-color-modal", "is_open"),
        Output("primary-open-color-modal", "style")
     ],
    [
        Input("primary-open-color-modal", "n_clicks"),
        Input("primary-apply-color-modal", "n_clicks"),
        #Input("select-color", "n_clicks"),
    ],
    [
        State("primary-color-modal", "is_open"),
        State("primary-color-picker", "value")

    ],
    prevent_initial_call=True
)
def _toggle_primary_color_modal(n1, n2, is_open, color_value):
    tid = ctx.triggered_id
    if tid == "primary-open-color-modal":
        return not is_open, dash.no_update
    elif tid == "primary-apply-color-modal":
        rgb = color_value["rgb"]
        r, g, b = rgb["r"], rgb["g"], rgb["b"]
        color = f"rgb({r}, {g}, {b})"
        style = {"width": "100%", "height": "40px", "background-color": color}
    else:
        raise ValueError("")
    return not is_open, style

@app.callback(
    [
        Output("secondary-color-modal", "is_open"),
        Output("secondary-open-color-modal", "style")
     ],
    [
        Input("secondary-open-color-modal", "n_clicks"),
        Input("secondary-apply-color-modal", "n_clicks"),
        #Input("select-color", "n_clicks"),
    ],
    [
        State("secondary-color-modal", "is_open"),
        State("secondary-color-picker", "value")

    ],
    prevent_initial_call=True
)
def _toggle_secondary_color_modal(n1, n2, is_open, color_value):
    tid = ctx.triggered_id
    if tid == "secondary-open-color-modal":
        return not is_open, dash.no_update
    elif tid == "secondary-apply-color-modal":
        rgb = color_value["rgb"]
        r, g, b = rgb["r"], rgb["g"], rgb["b"]
        color = f"rgb({r}, {g}, {b})"
        style = {"width": "100%", "height": "40px", "background-color": color}
    else:
        raise ValueError("")
    return not is_open, style


@app.callback(
    Output("download-image", "data"),
    [
        Input("download-image-button", "n_clicks"),

    ],
    [
        State("named-download", "value"),
        State("protein-id", "children"),
        State("replicate-mode", "on"),
        State("primary-open-color-modal", "style"),
        State("secondary-open-color-modal", "style"),
    ],
    prevent_initial_call=True
)
def _download_image(n_clicks, filename, key, replicate_mode, primary_color, secondary_color):
    key = key.split("Protein ")[-1]
    colors = primary_color['background-color'], secondary_color['background-color']


    filename = os.path.basename(filename)
    array, _ = rdpmsdata[key]
    i = 0
    if rdpmsdata.current_kernel_size is not None:
        i = int(np.floor(rdpmsdata.current_kernel_size / 2))
    if replicate_mode:
        fig = plot_replicate_distribution(array, rdpmsdata.internal_design_matrix, groups="RNAse", offset=i, colors=colors)
    else:
        fig = plot_distribution(array, rdpmsdata.internal_design_matrix, groups="RNAse", offset=i, colors=colors)
    fig.layout.template = "plotly_white"
    fig.update_layout(
        font=dict(color="black"),
        yaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),
        xaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),

    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ))
    fig.update_layout(
        margin={"t": 0, "b": 30, "r": 50},
        font=dict(
            size=16,
        )
    )
    fig.update_xaxes(dtick=1)
    tmpfile = os.path.join(TMPDIR.name, filename)
    fig.write_image(tmpfile)
    assert os.path.exists(tmpfile)
    return dcc.send_file(tmpfile)



clientside_callback(
    ClientsideFunction(
        namespace="clientside",
        function_name="function1"

    ),
    [Output("placeholder", "children")],
    [
        Input("night-mode", "on"),
        Input("secondary-open-color-modal", "style"),
    ],
)

clientside_callback(
    ClientsideFunction(
        namespace="clientside",
        function_name="nightMode"

    ),
    [Output("placeholder2", "children")],
    [
        Input("night-mode", "on"),
    ],
)


def _gui_wrapper(args):
    gui_wrapper(args.input, args.design_matrix, args.sep, args.logbase, args.debug, args.port, args.host)


def gui_wrapper(input, design_matrix, sep, logbase, debug, port, host):
    global rdpmsdata
    global data
    rdpmsdata = RDPMSpecData.from_files(input, design_matrix, sep=sep, logbase=logbase)
    data = rdpmsdata.df
    _get_app_layout(app)
    app.run(debug=debug, port=port, host=host)


if __name__ == '__main__':
    file = os.path.abspath("../../testData/testFile.tsv")
    assert os.path.exists(file)
    df = pd.read_csv(file, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    design = pd.read_csv(os.path.abspath("../../testData/testDesign.tsv"), sep="\t")
    rdpmsdata = RDPMSpecData(df, design, logbase=2)
    data = rdpmsdata.df







    _get_app_layout(app)
    app.run(debug=False, port=8080, host="127.0.0.1")
