import os

import dash_bootstrap_components as dbc

from RAPDOR.visualize.staticContent import LOGO, LIGHT_LOGO
assert os.path.exists(LOGO), f"{LOGO} does not exist"
assert os.path.exists(LIGHT_LOGO), f"{LIGHT_LOGO} does not exist"
from dash_extensions.enrich import DashProxy, Output, Input, State, Serverside, html, dcc, \
    ServersideOutputTransform, FileSystemBackend, clientside_callback, ClientsideFunction, RedisBackend
from RAPDOR.visualize import DISPLAY, DISPLAY_FILE, CONFIG
from RAPDOR.visualize.backends import background_callback_manager, celery_app, data_backend

FILEDIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(FILEDIR, "assets")



app = DashProxy(
    "RAPDOR Dashboard",
    title="RAPDOR Visualizer",
    external_stylesheets=[dbc.themes.DARKLY, "https://use.fontawesome.com/releases/v6.5.1/css/all.css"],
    assets_folder=ASSETS_DIR,
    index_string=open(os.path.join(ASSETS_DIR, "index.html")).read(),
    prevent_initial_callbacks="initial_duplicate",
    transforms=[ServersideOutputTransform(backends=[data_backend])],
    use_pages=True,
    pages_folder=os.path.join(FILEDIR, "pages"),
    background_callback_manager=background_callback_manager
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
        Input("primary-color", "data"),
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
        function_name="styleTutorial",

    ),
    [Output("placeholder8", "children")],
    [
        Input("primary-color", "data"),
        Input("secondary-color", "data"),
    ],
    [
        State("t-fill-start", "data"),
        State("t-black-start", "data")
    ]
)

clientside_callback(
    ClientsideFunction(
        namespace="clientside",
        function_name="activateTutorial" if not DISPLAY else "activateDisplayTutorial",

    ),
    [Output("placeholder9", "children")],
    [
        Input("tut-btn", "n_clicks"),
        Input("tut-end", "n_clicks"),
        Input("url", "pathname"),
        State("tutorial-dialog", "data")
    ],
    prevent_initial_call=True,

)

clientside_callback(
    ClientsideFunction(
        namespace="clientside",
        function_name="tutorialStep",

    ),
    [Output("tut-output", "data")],
    [
        Input("tut-next", "n_clicks"),
        Input("tut-prev", "n_clicks"),
    ],
    prevent_initial_call=True,

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

clientside_callback(
    ClientsideFunction(
        namespace="clientside",
        function_name="displayToolTip",

    ),
    [Output("placeholder7", "children")],
    [
        Input("tbl", "data"),
    ],
)
