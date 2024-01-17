import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import dcc, html

from RAPDOR.visualize import BOOTSROW, BOOTSH5
from RAPDOR.plots import COLOR_SCHEMES, DEFAULT_COLORS


def _color_theme_modal(nr: int = 1):
    modal = dbc.Modal(
        [
            dbc.ModalHeader(f"Select Color Scheme"),
            dbc.ModalBody(
                [
                    dcc.Dropdown(
                        list(COLOR_SCHEMES.keys()),
                        "Flamingo",
                        id=f"color-scheme-dropdown-{nr}",
                        clearable=False
                    )
                ]
            ),
            dbc.ModalFooter(
                dbc.Button("Apply", id=f"apply-color-scheme-{nr}", className="ml-auto",
                           n_clicks=0)),
        ],
        id=f"color-scheme-modal-{nr}",
    )
    return modal


def _modal_color_selection(number):
    color = DEFAULT_COLORS[number.split("-")[0]]
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


def _color_selection():
    div = html.Div(
        [
            html.Div(
                [
                    html.H5(
                        [
                            "Color Scheme",
                            html.I(className="fas fa-question-circle fa px-2", id="color-select-tip"),
                            dbc.Tooltip("Click on the colored box to select custom colors",
                                        target="color-select-tip"),
                        ]
                    ),

                ],
                className=BOOTSH5),

            html.Div(
                dcc.Dropdown(
                    list(COLOR_SCHEMES.keys()),
                    None,
                    id=f"color-scheme-dropdown",
                    persistence=True,
                    persistence_type="session"
                ),
                className="col-6 col-md-4"
            ),
            html.Div(
                html.Button(
                    '', id='primary-2-open-color-modal', n_clicks=0, className="btn primary-color-btn",
                    style={"width": "100%", "height": "100%"}
                ),
                className="col-3 col-md-4 d-flex align-items-center"
            ),
            html.Div(
                html.Button(
                    '', id='secondary-2-open-color-modal', n_clicks=0,
                    className="btn secondary-color-btn",
                    style={"width": "100%", "height": "100%"}
                ),
                className="col-3 col-md-4 d-flex align-items-center"
            ),

        ],

        className=BOOTSROW
                        )
    return div
