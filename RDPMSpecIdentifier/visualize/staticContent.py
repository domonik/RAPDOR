import base64
import logging
import os
import re

import dash_daq as daq
from dash import html
from dash_extensions.enrich import page_registry, State, Input, Output, callback
from dash import dcc
import dash_bootstrap_components as dbc

import RDPMSpecIdentifier

FILEDIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(FILEDIR, "assets")
LOGO = os.path.join(ASSETS_DIR, "RDPMSpecIdentifier_dark_no_text.svg")
LIGHT_LOGO = os.path.join(ASSETS_DIR, "RDPMSpecIdentifier_light_no_text.svg")
encoded_img = base64.b64encode(open(LOGO, 'rb').read())
IMG_TEXT = open(LOGO, 'r').read()
color = "fill:#ff8add"
res = re.search(color, IMG_TEXT)
COLOR_IDX = res.start()
COLOR_END =res.end()

res2 = re.search("fill:#f2f2f2", IMG_TEXT)
BS = res2.start()
BE = res2.end()

logger = logging.getLogger("RDPMSpecIdentifier")
DEFAULT_COLORS = {"primary": "rgb(138, 255, 172)", "secondary": "rgb(255, 138, 221)"}
COLOR_SCHEMES = {
    "Flamingo": (DEFAULT_COLORS["primary"], DEFAULT_COLORS["secondary"]),
    "Viking": ("rgb(79, 38, 131) ", "rgb(255, 198, 47)"),
    "Dolphin": ("rgb(0, 142, 151) ", "rgb(252, 76, 2)"),
    "Cardinal": ("rgb(151,35,63)", "rgb(0,0,0)"),
    "Falcon": ("rgb(167, 25, 48)", "rgb(0, 0, 0)"),
    "Raven": ("rgb(26, 25, 95)", "rgb(0, 0, 0)"),
    "Bill": ("rgb(0, 51, 141)", "rgb(198, 12, 48)"),
    "Panther": ("rgb(0, 133, 202)", "rgb(16, 24, 32)"),
    "Bear": ("rgb(111, 22, 42)", "rgb(12, 35, 64)"),
    "Bengal": ("rgb(251, 79, 20)", "rgb(0, 0, 0)"),
    "Brown": ("rgb(49, 29, 0)", "rgb(255, 60, 0)"),
    "Cowboy": ("rgb(0, 34, 68)", "rgb(255, 255, 255)"),
    "Bronco": ("rgb(251, 79, 20)", "rgb(0, 34, 68)"),
    "Lion": ("rgb(0, 118, 182)", "rgb(176, 183, 188)"),
    "Packer": ("rgb(24, 48, 40)", "rgb(255, 184, 28)"),
    "Texan": ("rgb(3, 32, 47)", "rgb(167, 25, 48)"),
    "Colt": ("rgb(0, 44, 95)", "rgb(162, 170, 173)"),
    "Jaguar": ("rgb(215, 162, 42)", "rgb(0, 103, 120)"),
    "Chief": ("rgb(227, 24, 55)", "rgb(255, 184, 28)"),
    "Charger": ("rgb(0, 128, 198)", "rgb(255, 194, 14)"),
    "Ram": ("rgb(0, 53, 148)", "rgb(255, 163, 0)"),
    "Patriot": ("rgb(0, 34, 68)", "rgb(198, 12, 48)"),
    "Saint": ("rgb(211, 188, 141)", "rgb(16, 24, 31)"),
    "Giant": ("rgb(100, 75, 0, 30)", "rgb(163, 13, 45)"),
    "Jet": ("rgb(18, 87, 64)", "rgb(255, 255, 255)"),
    "Raider": ("rgb(0, 0, 0)", "rgb(165, 172, 175)"),
    "Eagle": ("rgb(0, 76, 84)", "rgb(165, 172, 175)"),
    "Steeler": ("rgb(255, 182, 18)", "rgb(16, 24, 32)"),
    "49": ("rgb(170, 0, 0)", "rgb(173, 153, 93)"),
    "Seahawk": ("rgb(0, 34, 68)", "rgb(105, 190, 40)"),
    "Buccaneer": ("rgb(213, 10, 10)", "rgb(255, 121, 0)"),
    "Titan": ("rgb(75, 146, 219)", "rgb(200, 16, 46)"),
    "Commander": ("rgb(90, 20, 20)", "rgb(255, 182, 18)"),
}


def _header_layout():
    svg = 'data:image/svg+xml;base64,{}'.format(encoded_img.decode())
    header = html.Div(
        html.Div(
            html.Div(
                [
                    dbc.Offcanvas(
                        dbc.ListGroup(
                            [
                                dbc.ListGroupItem(page["name"], href=page["path"])
                                for page in page_registry.values()
                                if page["module"] != "pages.not_found_404"
                            ] + [
                                dbc.ListGroupItem(
                                    "Help",
                                    href="https://rdpmspecidentifier.readthedocs.io/en/latest/dashboard.html"
                                )
                            ]
                        ),
                        id="offcanvas",
                        is_open=False,
                    ),
                    dcc.Store(id="fill-start", data=COLOR_IDX),
                    dcc.Store(id="black-start", data=BS),
                    html.Div(
                        html.Button("Pages", id="open-offcanvas", n_clicks=0, className="align-self-start pages-btn"),
                        className="col-2 d-lg-none d-flex align-items-center"
                    ),
                    html.Div([
                        dcc.Link("Upload", href="/", className="px-2"),
                        dcc.Link("Analysis", href="/analysis", className="px-2"),
                        dcc.Link("Figure Factory", href="/figure_factory", className="px-2", style={"white-space": "nowrap"}),
                        dcc.Link("Help", href="https://rdpmspecidentifier.readthedocs.io/en/latest/dashboard.html",
                                 className="px-2", target="_blank"),
                        ],
                        className=" col-3 d-lg-flex d-none align-items-center"
                    ),
                    html.Div(
                        html.Img(src=svg, style={"width": "20%", "min-width": "300px"}, className="p-1",
                                 id="flamingo-svg"),
                        className="col-md-6 col-9 justify-content-center justify-conent-md-start", id="logo-container"
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
            className="databox header-box p-2",
            style={"text-align": "center"},
        ),
        className="col-12 m-0 px-0 pb-1 justify-content-center"
    )
    return header

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
                    className="text-end"),
                html.P(
                    html.A(
                        f"Help",
                        className="text-end",
                        href="https://rdpmspecidentifier.readthedocs.io/en/latest/",
                        target="_blank"
                    ),
                    className="text-end")
            ],
            className="col-12 col-md-4 flex-column justify-content-end align-items-end"
        )
    ]
    return footer


VERSION = RDPMSpecIdentifier.__version__

