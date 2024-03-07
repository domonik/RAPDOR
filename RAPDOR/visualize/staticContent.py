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
IMG_TEXT = open(LOGO, 'r').read().replace("\n", "").replace("\r\n", "")
color = "fill:#ff8add"
res = re.finditer(color, IMG_TEXT)
COLOR_IDX = [result.start() for result in res]
c2 = "fill:#8affac"
res2 = re.finditer(c2, IMG_TEXT)
C2STARTS = [result.start() for result in res2]
encoded_img = base64.b64encode(IMG_TEXT.encode())

TUTORIALRAPDOR = os.path.join(ASSETS_DIR, "RAPDORRaptor.svg")
TUTORIALIMGTEXT = open(TUTORIALRAPDOR, 'r').read().replace("\n", "").replace("\r\n", "")
ENCODEDTUTIMG = base64.b64encode(TUTORIALIMGTEXT.encode())
res = re.finditer(color, TUTORIALIMGTEXT)
TCSTARTS = [result.start() for result in res]
res2 = re.finditer(c2, TUTORIALIMGTEXT)
TC2STARTS = [result.start() for result in res2]


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
                                    "Docs",
                                    href="https://domonik.github.io/RAPDOR/"
                                )
                            ]
                        ),
                        id="offcanvas",
                        is_open=False,
                    ),
                    dcc.Store(id="fill-start", data=COLOR_IDX),
                    dcc.Store(id="black-start", data=C2STARTS),
                    dcc.Store(id="t-fill-start", data=TCSTARTS),
                    dcc.Store(id="t-black-start", data=TC2STARTS),
                    html.Div(
                        html.Button(id="open-offcanvas", n_clicks=0,
                                    className="align-self-start pages-btn fa fa-bars fa-xl"),
                        className="col-1 d-lg-none d-flex align-items-center"
                    ),
                    html.Div([
                                 dcc.Link(page["name"], href=page["path"], className="px-2", style={"white-space": "nowrap"}) for page in
                                 page_registry.values()
                                 if page["module"] != "pages.not_found_404"
                             ] + [
                                 dcc.Link("Docs", href="https://domonik.github.io/RAPDOR/",
                                 className="px-2", target="_blank"),
                        ],
                        className=" col-3 d-lg-flex d-none align-items-center"
                    ),
                    html.Div(
                        html.Img(src=svg, style={"width": "20%", "min-width": "150px"}, className="p-0",
                                 id="flamingo-svg"),
                        className="col-md-6 col-7 justify-content-center justify-conent-md-start", id="logo-container"
                    ),
                    html.Div(
                        [
                            daq.BooleanSwitch(
                            label='',
                            labelPosition='left',
                            color="var(--r-text-color)",
                            on=True,
                            id="night-mode",
                            className="align-self-center px-2",
                            persistence=True

                        ),

                        ],
                        className="col-1  d-flex justify-content-end justify-self-end align-items-center"
                    ),
                    html.Div(
                        [
                            html.Span("Tutorial", style={"font-family": "Font Awesome 6 Free", "color": "white"}, className="d-lg-flex d-none"),
                            html.Button(id="tut-btn", style={"z-index": "10000"},
                                    className="p-2 btn-secondary fa fa-regular fa-circle-play fa-2xl", )
                        ],
                        className="col-2 justify-content-end justify-self-end d-flex align-items-center"

                    )



                ],
                className="row px-1 justify-content-between"
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
        ),


    ]
    return footer


def _tutorial_raptor():
    svg = 'data:image/svg+xml;base64,{}'.format(ENCODEDTUTIMG.decode())
    content = [

        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    html.H5("Tutorial", id="tut-head"), className="col-8 p-1 align-items-center d-flex"
                                ),
                                html.Div(
                                    html.Button("", className="btn-tut fa fa-xmark",
                                                style={"pointer-events": "all"}, id="tut-end"
                                                ),
                                    className="justify-self-end d-flex p-1 col-2 justify-content-end"
                                ),


                            ],

                            className="row justify-content-between px-4 pt-1"

                        ),
                        html.Div(

                            [
                                html.Div("", id="tut-text", className="col-12 p-2")
                            ],
                            className="row px-4"
                        ),
                        html.Div(
                            [
                                html.Div(html.Button(className="btn-tut fa fa-angle-left fa-lg"),  id="tut-prev", className="col-2 d-flex p-1 justify-content-start"),
                                html.Div(html.Span("1/15"),  id="tut-step", className="col-2 d-flex p-1 justify-content-center"),
                                html.Div(html.Button(className="btn-tut fa fa-angle-right fa-lg"), id="tut-next",  className="col-2 d-flex p-1 justify-content-end")
                            ],
                            className="row justify-content-between px-4 pb-1"
                        ),

                    ],

                    className="dialog-1",
                    style={"pointer-events": "all"}

                ),

            ],

            className="col-lg-4 col-12 align-self-center"
        ),
        html.Div(
            html.Img(src=svg, className="tutorial-rapdor-svg p-0",
                     id="tutorial-rapdor-svg", ),
            className="tutorial-rapdor col-lg-3 d-none d-lg-flex p-0 m-0", id="TutorialRapdor"
        ),

    ]
    return content


VERSION = RAPDOR.__version__

