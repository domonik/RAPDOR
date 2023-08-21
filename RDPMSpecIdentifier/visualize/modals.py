import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import html

from RDPMSpecIdentifier.visualize.staticContent import DEFAULT_COLORS


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
