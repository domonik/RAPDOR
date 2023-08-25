import base64
import logging
import os
import re

import dash_daq as daq
from dash import html
from dash import dcc

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



def _header_layout():
    svg = 'data:image/svg+xml;base64,{}'.format(encoded_img.decode())
    header = html.Div(
        html.Div(
            html.Div(
                [
                    dcc.Store(id="fill-start", data=COLOR_IDX),
                    dcc.Store(id="black-start", data=BS),
                    html.Div(className="col-md-3 col-0"),
                    html.Div(
                        html.Img(src=svg, style={"width": "20%", "min-width": "300px"}, className="p-1",
                                 id="flamingo-svg"),
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


VERSION = RDPMSpecIdentifier.__version__
