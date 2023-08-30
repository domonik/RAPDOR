import base64
import pickle
import re

import dash
from dash import Output, Input, html, ctx, dcc
from dash.exceptions import PreventUpdate

from RDPMSpecIdentifier.visualize.appDefinition import app
from RDPMSpecIdentifier.visualize.staticContent import IMG_TEXT, COLOR_IDX, COLOR_END, BS, BE
from dash_extensions.enrich import Serverside, State
from RDPMSpecIdentifier.datastructures import RDPMSpecData
import pandas as pd
import uuid
from tempfile import TemporaryDirectory
import os
import RDPMSpecIdentifier.visualize as rdv
import logging

logger = logging.getLogger(__name__)

@app.callback(
    Output("data-store", "data", allow_duplicate=True),
    Output("kernel-slider", "value"),
    Output("distance-method", "value"),
    Output("unique-id", "data"),
    Output('cluster-feature-slider', 'value'),
    Input("unique-id", "data"),
    State("data-initial-store", "data"),
    State("data-store", "data"),

)
def load_state(uid, data, saved):
    logger.info("Loading Data")
    if uid is None:
        uid = str(uuid.uuid4())
    if saved is None:
        logger.info("Loading from initial state")
        rdpmspec = RDPMSpecData.from_json(data)
    else:
        logger.info("Loading from saved state")
        rdpmspec = saved
    kernel = rdpmspec.state.kernel_size if rdpmspec.state.kernel_size is not None else 3
    dm = rdpmspec.state.distance_method if rdpmspec.state.distance_method is not None else dash.no_update
    cf_slider = rdpmspec.state.cluster_kernel_distance if rdpmspec.state.cluster_kernel_distance is not None else dash.no_update
    return Serverside(rdpmspec, key=uid), kernel, dm, uid, cf_slider


@app.callback(
    Output("recomputation", "children"),
    Output("data-store", "data"),
    Input("kernel-slider", "value"),
    Input("distance-method", "value"),
    State("data-store", "data"),
    State("unique-id", "data"),
    prevent_initial_call=True
)
def recompute_data(kernel_size, distance_method, rdpmsdata, uid):
    logger.info("Normalization")
    eps = 0 if distance_method == "Jensen-Shannon-Distance" else 10  # Todo: Make this optional
    rdpmsdata: RDPMSpecData
    if rdpmsdata.state.kernel_size != kernel_size or rdpmsdata.state.distance_method != distance_method:
        rdpmsdata.normalize_and_get_distances(method=distance_method, kernel=kernel_size, eps=eps)
        return html.Div(), Serverside(rdpmsdata, key=uid)
    logger.info("Data already Normalized")
    raise PreventUpdate

#
# @app.callback(
#     Output("logo-container", "children"),
#     Input("night-mode", "on"),
#     Input("secondary-color", "data"),
# )
# def update_logo(night_mode, color):
#     rep = f"fill:{color}"
#     l_image_text = IMG_TEXT[:COLOR_IDX] + rep + IMG_TEXT[COLOR_END:]
#     if not night_mode:
#         l_image_text = re.sub("fill:#f2f2f2", "fill:black", l_image_text)
#     encoded_img = base64.b64encode(l_image_text.encode())
#     img = 'data:image/svg+xml;base64,{}'.format(encoded_img.decode())
#     return html.Img(src=img, style={"width": "20%", "min-width": "300px"}, className="p-1"),


@app.callback(
        Output("protein-id", "children"),
        Output("current-protein-id", "data"),

    [
        Input('tbl', 'active_cell'),
        Input("test-div", "children")
    ],
    State("data-store", "data")
)
def update_selected_id(active_cell, test_div, rdpmsdata):
    logger.info("Updating Selected Protein")
    if ctx.triggered_id == "tbl":
        if active_cell is None:
            raise PreventUpdate
        active_row_id = active_cell["row_id"]
        protein = rdpmsdata.df.loc[active_row_id, "RDPMSpecID"]
        protein = f"Protein {protein}"
    elif ctx.triggered_id == "test-div":
        active_row_id = int(test_div)
        protein = rdpmsdata.df.loc[active_row_id, "RDPMSpecID"]

    else:
        raise PreventUpdate

    return protein, active_row_id


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export-btn", "n_clicks"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def download_dataframe(n_clicks, rdpmsdata):
    return dcc.send_data_frame(rdpmsdata.extra_df.to_csv, "RDPMSpecIdentifier.tsv", sep="\t")




@app.callback(
    Output("download-pickle", "data"),
    Input("export-pickle-btn", "n_clicks"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def download_json(n_clicks, rdpmsdata):
    ret_val = dict(
        content=rdpmsdata.to_jsons(),
        filename="RDPMSpecIdentifier.json"
    )

    return ret_val
