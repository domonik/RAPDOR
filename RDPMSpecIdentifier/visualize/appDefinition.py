import os

import dash_bootstrap_components as dbc
from dash import clientside_callback, ClientsideFunction

from RDPMSpecIdentifier.visualize.staticContent import LOGO, LIGHT_LOGO
assert os.path.exists(LOGO), f"{LOGO} does not exist"
assert os.path.exists(LIGHT_LOGO), f"{LIGHT_LOGO} does not exist"
from dash_extensions.enrich import DashProxy, Output, Input, State, Serverside, html, dcc, \
    ServersideOutputTransform, FileSystemBackend
from RDPMSpecIdentifier.visualize import DISPLAY, DISPLAY_FILE
from RDPMSpecIdentifier.visualize.dataBackEnd import DisplayModeBackend

FILEDIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(FILEDIR, "assets")

if not DISPLAY:
    another_backend = FileSystemBackend("file_system_backend", threshold=200)
else:
    another_backend = DisplayModeBackend(DISPLAY_FILE)

app = DashProxy(
    "RDPMSpecIdentifier Dashboard",
    title="RDPMSpec Visualizer",
    external_stylesheets=[dbc.themes.DARKLY, "https://use.fontawesome.com/releases/v5.10.2/css/all.css"],
    assets_folder=ASSETS_DIR,
    index_string=open(os.path.join(ASSETS_DIR, "index.html")).read(),
    prevent_initial_callbacks="initial_duplicate",
    transforms=[ServersideOutputTransform(backends=[another_backend])],
    use_pages=True,
    pages_folder=os.path.join(FILEDIR, "pages")
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

clientside_callback(
    ClientsideFunction(
        namespace="clientside",
        function_name="styleSelectedTableRow",

    ),
    [Output("placeholder4", "children")],
    [
        Input("protein-id", "children"),
        Input("tbl", "data"),
    ],
)

clientside_callback(
    ClientsideFunction(
        namespace="clientside",
        function_name="restyleRadio",

    ),
    [Output("placeholder5", "children")],
    [
        Input("plot-type-radio-ff", "options"),
    ],
)

clientside_callback(
    ClientsideFunction(
        namespace="clientside",
        function_name="moveBtn",

    ),
    [Output("placeholder6", "children")],
    [
        Input("tbl", "data"),
        Input("analysis-tabs", "value"),
    ],
)
