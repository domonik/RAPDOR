import base64
import logging
import os
import re

import dash_daq as daq
from dash import html
from dash_extensions.enrich import page_registry
from dash import dcc
import dash_bootstrap_components as dbc

import RAPDOR

FILEDIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(FILEDIR, "assets")
LOGO = os.path.join(ASSETS_DIR, "RAPDOR.svg")
LIGHT_LOGO = os.path.join(ASSETS_DIR, "RAPDOR.svg")
encoded_img = base64.b64encode(open(LOGO, 'rb').read())
IMG_TEXT = open(LOGO, 'r').read()
color = "fill:#ff8add"
res = re.finditer(color, IMG_TEXT)
COLOR_IDX = [result.start() for result in res]
c2 = "fill:#8affac"
res2 = re.finditer(c2, IMG_TEXT)
C2STARTS = [result.start() for result in res2]

logger = logging.getLogger("RAPDOR")


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
                                    href="https://RAPDOR.readthedocs.io/en/latest/dashboard.html"
                                )
                            ]
                        ),
                        id="offcanvas",
                        is_open=False,
                    ),
                    dcc.Store(id="fill-start", data=COLOR_IDX),
                    dcc.Store(id="black-start", data=C2STARTS),
                    html.Div(
                        html.Button("Pages", id="open-offcanvas", n_clicks=0, className="align-self-start pages-btn"),
                        className="col-2 d-lg-none d-flex align-items-center"
                    ),
                    html.Div([
                        dcc.Link("Upload", href="/", className="px-2"),
                        dcc.Link("Analysis", href="/analysis", className="px-2"),
                        dcc.Link("Figure Factory", href="/figure_factory", className="px-2", style={"white-space": "nowrap"}),
                        dcc.Link("Help", href="https://RAPDOR.readthedocs.io/en/latest/dashboard.html",
                                 className="px-2", target="_blank"),
                        ],
                        className=" col-3 d-lg-flex d-none align-items-center"
                    ),
                    html.Div(
                        html.Img(src=svg, style={"width": "20%", "min-width": "150px"}, className="p-0",
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
                        href="https://github.com/domonik/RAPDOR",
                        target="_blank"
                    ),
                    className="text-end"),
                html.P(
                    html.A(
                        f"Help",
                        className="text-end",
                        href="https://RAPDOR.readthedocs.io/en/latest/",
                        target="_blank"
                    ),
                    className="text-end")
            ],
            className="col-12 col-md-4 flex-column justify-content-end align-items-end"
        )
    ]
    return footer


VERSION = RAPDOR.__version__

