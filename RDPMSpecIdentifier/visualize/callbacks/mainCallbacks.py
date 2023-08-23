import base64
import re

from dash import Output, Input, html, ctx, dcc
from dash.exceptions import PreventUpdate

from RDPMSpecIdentifier.visualize.appDefinition import app
from RDPMSpecIdentifier.visualize.staticContent import IMG_TEXT, COLOR_IDX, COLOR_END
from dash_extensions.enrich import Serverside, State
from RDPMSpecIdentifier.datastructures import RDPMSpecData
import pandas as pd
import uuid





@app.callback(
    Output("recomputation", "children"),
    Output("data-store", "data"),
    Output("unique-id", "data"),
    Input("kernel-slider", "value"),
    Input("distance-method", "value"),
    State("data-store", "data"),
    State("design-store", "data"),
    State("intentity-store", "data"),
    State("logbase-store", "data"),
    State("unique-id", "data"),
)
def recompute_data(kernel_size, distance_method, data, design, intensities, logbase, uid):
    if uid is None:
        uid = str(uuid.uuid4())
    if data is None:
        intensities = pd.read_json(intensities)
        intensities.index = intensities.index.astype(str)
        design = pd.read_json(design)
        rdpmspec = RDPMSpecData(df=intensities, design=design, logbase=logbase)
    else:
        rdpmspec: RDPMSpecData = data
        method = rdpmspec.methods[distance_method]
        if rdpmspec.current_kernel_size == kernel_size and rdpmspec.current_method == method:
            return html.Div(), Serverside(rdpmspec, key=uid), uid

    method = rdpmspec.methods[distance_method]
    eps = 0 if distance_method == "Jensen-Shannon-Distance" else 10  # Todo: Make this optional
    rdpmspec.normalize_and_get_distances(method=method, kernel=kernel_size, eps=eps)
    return html.Div(), Serverside(rdpmspec, key=uid), uid


@app.callback(
    Output("logo-container", "children"),
    Input("night-mode", "on"),
    Input("secondary-open-color-modal", "style"),
)
def update_logo(night_mode, style):
    color2 = style["background-color"]
    rep = f"fill:{color2}"
    l_image_text = IMG_TEXT[:COLOR_IDX] + rep + IMG_TEXT[COLOR_END:]
    if not night_mode:
        l_image_text = re.sub("fill:#f2f2f2", "fill:black", l_image_text)
    encoded_img = base64.b64encode(l_image_text.encode())
    img = 'data:image/svg+xml;base64,{}'.format(encoded_img.decode())
    return html.Img(src=img, style={"width": "20%", "min-width": "300px"}, className="p-1"),


@app.callback(
        Output("protein-id", "children"),

    [
        Input('tbl', 'active_cell'),
        Input("test-div", "children")
    ],

)
def update_selected_id(active_cell, test_div):
    if ctx.triggered_id == "tbl":
        if active_cell is None:
            raise PreventUpdate
        active_row_id = active_cell["row_id"]
        active_row_id = f"Protein {active_row_id}"
    elif ctx.triggered_id == "test-div":
        active_row_id = f"Protein {test_div}"
    else:
        raise PreventUpdate

    return active_row_id


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export-btn", "n_clicks"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def download_dataframe(n_clicks, rdpmsdata):
    return dcc.send_data_frame(rdpmsdata.extra_df.to_csv, "RDPMSpecIdentifier.tsv", sep="\t")
