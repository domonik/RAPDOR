import dash_bootstrap_components as dbc
from dash import html, dcc

KMEANS_ARGS = 2
DBSCAN_ARGS = 2
HDBSCAN_ARGS = 2

def _modal_image_download():
    modal = dbc.Modal(
        [
            dbc.ModalHeader("Select Download Options"),
            dbc.ModalBody(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    _get_download_input("download-name", "Filename", "text"),
                                    _get_download_input("download-width", "Width [px]", "number", 800),
                                    _get_download_input("download-height", "Height [px]", "number", 500),
                                ],
                                className="col-12 col-md-5 mx-2 my-0"
                            ),
                            html.Div(
                                [
                                    _get_download_input("download-marker-size", "Marker size", "number", 7),
                                    _get_download_input("download-line-width", "Line Width", "number", 3),
                                    _get_download_input("download-grid-width", "Grid Width", "number", 1),
                                ],
                                className="col-12 col-md-5 mx-2 my-0"
                            ),
                            html.Div(
                                [
                                    dbc.RadioItems(
                                        options=[
                                            {'label': 'Distribution', 'value': 0},
                                            {'label': 'Heatmap', 'value': 1},
                                            {'label': 'Westernblot', 'value': 2},
                                        ],
                                        value=0,
                                        inline=True,
                                        className="d-flex justify-content-between radio-items",
                                        labelCheckedClassName="checked-radio-text",
                                        inputCheckedClassName="checked-radio-item",
                                        id="plot-type-radio",
                                    ),
                                ],
                                className="col-10 my-2"
                            ),
                            html.Div(className="col-12 col-md-7 mx-2 my-0"),
                            html.Div(
                                dbc.Button("Download", id="download-image-button",
                                           className="btn btn-primary col-12 "),
                                className="col-12 col-md-3 mx-2 my-0"),


                        ], className="row justify-content-center"
                    ),


                    html.Div(
                        html.H4("Preview", className="col-10 col-md-6 mt-1"),
                        className="row justify-content-center"
                    ),
                    html.Div(
                        [
                            html.Div(
                                [],
                                className="col-12 col-md-11 m-2", id="distribution-graph-download-preview",
                                style={
                                    "overflow-x": "auto",
                                    "background-color": "white",
                                    "border-radius": "10px"
                                }
                            ),
                        ],
                        className="row justify-content-around",
                    ),

                ]
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="close", className="ml-auto",
                           n_clicks=0)),
        ],
        id="modal",
        size="xl"
    )
    return modal


def _modal_cluster_image_download():
    modal = dbc.Modal(
        [
            dbc.ModalHeader("Select file Name"),
            dbc.ModalBody(
                [
                    html.Div(
                        [
                            html.Div(dbc.Input("cluster-download",),
                                        className=" col-9"),
                            dbc.Button("Download", id="download-cluster-image-button", className="btn btn-primary col-3"),
                        ],
                        className="row justify-content-around",
                    )
                ]
            ),
        ],
        id="cluster-img-modal",
    )
    return modal


def _modal_hdbscan_cluster_settings():
    name = "HDBSCAN"
    modal = dbc.Modal(
        [
            dbc.ModalHeader(f"{name} Settings"),
            dbc.ModalBody(
                [
                    _get_arg_input(name, "min_cluster_size", "number", 12),
                    _get_arg_input(name, "cluster_selection_epsilon", "number", 0.0),
                ]
            ),
            dbc.ModalFooter(
                dbc.Button("Apply", id=f"{name}-apply-settings-modal", className="ml-auto",
                           n_clicks=0)),
        ],
        id=f"{name}-cluster-modal",
        fullscreen="md-down",
        size="lg"
    )
    return modal


def _modal_dbscan_cluster_settings():
    name = "DBSCAN"
    modal = dbc.Modal(
        [
            dbc.ModalHeader(f"{name} Settings"),
            dbc.ModalBody(
                [
                    _get_arg_input(name, "eps", "number", 0.5),
                    _get_arg_input(name, "min_samples", "number", 5),
                ]
            ),
            dbc.ModalFooter(
                dbc.Button("Apply", id=f"{name}-apply-settings-modal", className="ml-auto",
                           n_clicks=0)),
        ],
        id=f"{name}-cluster-modal",
        fullscreen="md-down",
        size="lg"
    )
    return modal

def _modal_kmeans_cluster_settings():
    name = "K-Means"
    modal = dbc.Modal(
        [
            dbc.ModalHeader(f"{name} Settings"),
            dbc.ModalBody(
                [
                    _get_arg_input(name, "n_clusters", "number", 8),
                    _get_arg_input(name, "random_state", "number", 0),
                ]
            ),
            dbc.ModalFooter(
                dbc.Button("Apply", id=f"{name}-apply-settings-modal", className="ml-auto",
                           n_clicks=0)),
        ],
        id=f"{name}-cluster-modal",
        fullscreen="md-down",
        size="lg"
    )
    return modal


def _get_arg_input(name, arg, d_type, default=None):
    div = html.Div(
        [
            html.Div(
                html.Span(arg, style={"text-align": "center"}),
                className="col-7 col-md-3 justify-content-center align-self-center"
            ),
            html.Div(
                dcc.Input(
                    style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white",
                           "text-align": "center"},
                    id=f"{name}-{arg}-input",
                    className="text-align-center",
                    value=default,
                    type=d_type,
                ),
                className="col-3 justify-content-center text-align-center"
            )
        ],
        className="row justify-content-around m-1",
    )
    return div

def _get_download_input(name, arg, d_type, default=None):
    div = html.Div(
        [
            html.Div(
                html.Span(arg, style={"text-align": "center"}),
                className="col-4 justify-content-center align-self-center"
            ),
            html.Div(
                dcc.Input(
                    style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white",
                           "text-align": "center"},
                    id=f"{name}-input",
                    className="text-align-center",
                    value=default,
                    type=d_type,
                ),
                className="col-8 justify-content-center text-align-center"
            )
        ],
        className="row justify-content-center p-1",
    )
    return div