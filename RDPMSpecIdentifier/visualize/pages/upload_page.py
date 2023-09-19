
import dash
from dash import html
import base64
from dash import dcc
from dash_extensions.enrich import callback, Input, Output, Serverside, State
from RDPMSpecIdentifier.datastructures import RDPMSpecData
import logging
import dash_bootstrap_components as dbc
import pandas as pd
from io import StringIO
from dash.exceptions import PreventUpdate
import os
from RDPMSpecIdentifier.visualize import DISABLED

logger = logging.getLogger(__name__)

dash.register_page(__name__, path='/')




def from_csv(disabled: bool = False):
    data = dcc.Tab(html.Div([
        html.Div(
            [
                html.Div(html.Span("Intensities"), className="col-3 justify-content-center align-self-center"),
                html.Div(
                    dcc.Upload(
                        id='upload-intensities',
                        children=html.Div(
                            'Drag and Drop or Select Files' if not disabled else "Upload disabled",
                            id="intensities-upload-text"
                        ),
                        className="text-align-center justify-text-center",
                        disabled=disabled,

                        style={
                            'width': '100%',
                            'borderWidth': '2px',
                            'height': "60px",
                            'lineHeight': '60px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            "borderColor": "var(--r-text-color)",
                            'textAlign': 'center',
                        },
                    ),
                    className="col-7 justify-content-center text-align-center"
                )

            ],
            className="row justify-content-center p-2"
        ),
        html.Div(
            [
                html.Div(html.Span("Design"),
                         className="col-3 justify-content-center align-self-center"),
                html.Div(
                    dcc.Upload(
                        id='upload-design',
                        children=html.Div(
                            'Drag and Drop or Select Files' if not disabled else "Upload disabled",
                            id="design-upload-text"
                        ),
                        className="text-align-center justify-text-center",
                        disabled=disabled,
                        style={
                            'width': '100%',
                            'height': "60px",
                            'lineHeight': '60px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            "borderColor": "var(--r-text-color)",
                            'textAlign': 'center',
                        },
                    ),
                    className="col-7 justify-content-center text-align-center"
                )

            ],
            className="row justify-content-center p-2"
        ),
        html.Div(
            [
                html.Div(html.Span("Log-Base"),
                         className="col-3 justify-content-center align-self-center"),
                html.Div(
                    dcc.Input(
                        style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white",
                               "text-align": "center"},
                        id="logbase",
                        className="text-align-center",
                        value=0,
                        type="number",
                        min=0,
                        disabled=disabled
                    ),
                    className="col-7 text-align-center align-items-center"
                ),
            ],
            className="row justify-content-center p-2"
        ),
        html.Div(
            [
                html.Div(html.Span("Seperator"),
                         className="col-3 justify-content-center align-self-center"),
                html.Div(
                    dbc.RadioItems(
                        options=[
                            {'label': 'Tab', 'value': '\t'},
                            {'label': 'Comma', 'value': ','},
                            {'label': 'Semicolon', 'value': ';'},
                        ],
                        value='\t',
                        inline=True,
                        className="d-flex justify-content-between radio-items",
                        labelCheckedClassName="checked-radio-text",
                        inputCheckedClassName="checked-radio-item",
                        id="seperator-radio",
                    ),
                    className="col-7"
                ),

            ],
            className="row justify-content-center p-2"
        ),
        html.Div(
            [
                html.Div(
                    html.Button("Upload", id="upload-csv-btn", className="btn btn-primary w-100", disabled=disabled),
                         className="col-10 justify-content-center align-self-center"
                ),
            ],
            className="row justify-content-center p-2"
        ),
    ], className="databox databox-open py-3"), label="From CSV", className="custom-tab", selected_className='custom-tab--selected')
    return data


def from_json(disabled):
    data = dcc.Tab(html.Div([
        html.Div(
            [
                html.Div(html.Span("JSON"),
                         className="col-3  justify-content-center align-self-center"),
                html.Div(
                    dcc.Upload(
                        id='upload-json',
                        children=html.Div(
                            'Drag and Drop or Select Files' if not disabled else "Upload disabled",
                            id="json-upload-text"
                        ),
                        className="text-align-center justify-text-center",
                        disabled=disabled,
                        style={
                            'width': '100%',
                            'height': "60px",
                            'lineHeight': '60px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            "borderColor": "var(--r-text-color)",
                            'textAlign': 'center',
                        },
                    ),
                    className="col-7 justify-content-center text-align-center"
                )

            ],
            className="row justify-content-center p-2"
        ),

    ], className="databox databox-open py-3"), label="From JSON", className="custom-tab",
        selected_className='custom-tab--selected'
    )
    return data


layout = html.Div(
    [
        html.Div(
            id="upload-alert-div",
            className="col-10"
        ),
        html.Div(
            html.Div(html.Div(html.H3("Import Data"), className="col-lg-7 col-xl-5 col-12 databox text-center"),
                     className="row justify-content-center"),
            className="col-12"
        ),
        html.Div(
            [
                dcc.Tabs(
                    [
                        from_csv(DISABLED),
                        from_json(DISABLED)
                    ],
                    parent_className='custom-tabs',
                    className='custom-tabs-container',
                )


            ],
            className="col-lg-7 col-12 col-xl-5 p-2"
        ),
    ], className="row p-2 justify-content-center"
)


@callback(
    Output("data-store", "data", allow_duplicate=True),
    Output("url", "pathname", allow_duplicate=True),
    Output("upload-alert-div", "children", allow_duplicate=True),
    Input("upload-json", "contents"),
    State("unique-id", "data")

)
def upload_json(data, uid):
    if data is None:
        return dash.no_update
    if uid is None:
        logger.error("User ID missing. Cannot assign Data without a user ID")
        return dash.no_update
    try:
        content = data.split(',')[1]
        decoded = base64.b64decode(content)
        rdpmspec = RDPMSpecData.from_json(decoded)
        rdpmspec = Serverside(rdpmspec, key=uid)
        redirect = "analysis"
        alert = []
    except Exception:
        rdpmspec = dash.no_update
        redirect = dash.no_update
        alert = html.Div(
            dbc.Alert(
                "Data is not in the expected format.",
                color="danger",
                dismissable=True,
            ),
            className="p-2 align-items-center, alert-msg",

        )
    return rdpmspec, redirect, alert


for name in ("intensities", "design", "json"):
    @callback(
        Output(f"{name}-upload-text", "children"),
        Input(f"upload-{name}", "filename")
    )
    def change_intensity_text(filename):
        if filename is None:
            raise PreventUpdate
        return filename




@callback(
    Output("data-store", "data", allow_duplicate=True),
    Output("url", "pathname", allow_duplicate=True),
    Output("upload-alert-div", "children", allow_duplicate=True),
    Input("upload-csv-btn", "n_clicks"),
    State("unique-id", "data"),
    State("seperator-radio", "value"),
    State("upload-intensities", "contents"),
    State("upload-design", "contents"),
    State("logbase", "value"),
    prevent_initial_call=True
)
def upload_from_csv(btn, uid, sep, intensities_content, design_content, logbase):
    if intensities_content is None and design_content is None:
        raise PreventUpdate
    try:
        intensities_content = intensities_content.split(",")[1]
        intensities_content = base64.b64decode(intensities_content).decode()
        design_content = design_content.split(",")[1]
        design_content = base64.b64decode(design_content).decode()
        df = pd.read_csv(StringIO(intensities_content), sep=sep)
        design = pd.read_csv(StringIO(design_content), sep=sep)
        rdpmsdata = RDPMSpecData(df, design, logbase=None if logbase == 0 else logbase)
        rdpmsdata = Serverside(rdpmsdata, key=uid)
        redirect = "analysis"
        alert = []
    except Exception as e:
        rdpmsdata = dash.no_update
        redirect = dash.no_update
        alert = html.Div(
            dbc.Alert(
                "Data is not in the expected format.",
                color="danger",
                dismissable=True,
            ),
            className="p-2 align-items-center, alert-msg",

        )
    return rdpmsdata, redirect, alert
