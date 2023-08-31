import os

import dash_bootstrap_components as dbc
import plotly.io as pio
from dash import clientside_callback, ClientsideFunction

from RDPMSpecIdentifier.visualize.staticContent import LOGO, LIGHT_LOGO

assert os.path.exists(LOGO), f"{LOGO} does not exist"
assert os.path.exists(LIGHT_LOGO), f"{LIGHT_LOGO} does not exist"
from dash_extensions.enrich import DashProxy, Output, Input, State, Serverside, html, dcc, \
    ServersideOutputTransform, FileSystemBackend

FILEDIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(FILEDIR, "assets")

another_backend = FileSystemBackend("file_system_backend", threshold=200)

app = DashProxy(
    "RDPMSpecIdentifier Dashboard",
    title="RDPMSpec Visualizer",
    external_stylesheets=[dbc.themes.DARKLY],
    assets_folder=ASSETS_DIR,
    index_string=open(os.path.join(ASSETS_DIR, "index.html")).read(),
    prevent_initial_callbacks="initial_duplicate",
    transforms=[ServersideOutputTransform(backends=[another_backend])],
    use_pages=True,
    pages_folder=os.path.join(FILEDIR, "pages")
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


clientside_callback(
    ClientsideFunction(
        namespace="clientside",
        function_name="nightMode"

    ),
    [Output("placeholder2", "children")],
    [
        Input("night-mode", "on"),
        Input("secondary-color", "data"),
        Input("primary-color", "data"),

    ],
)


clientside_callback(
    ClientsideFunction(
        namespace="clientside",
        function_name="styleFlamingo",

    ),
    [Output("placeholder3", "children")],
    [
        Input("night-mode", "on"),
        Input("secondary-color", "data"),
    ],
    [
        State("fill-start", "data"),
        State("black-start", "data")
    ]
)
