import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import html, dcc
from RDPMSpecIdentifier.visualize.staticContent import DEFAULT_COLORS

KMEANS_ARGS = 2
DBSCAN_ARGS = 2
HDBSCAN_ARGS = 2

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


def _modal_hdbscan_cluster_settings():
    name = "HDBSCAN"
    modal = dbc.Modal(
        [
            dbc.ModalHeader(f"{name} Settings"),
            dbc.ModalBody(
                [
                    _get_arg_input(name, "min_cluster_size", "number", 3),
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
        className="row justify-content-around",
    )
    return div

