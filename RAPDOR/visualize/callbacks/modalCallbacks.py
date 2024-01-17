import os

import dash
import numpy as np
from dash import Output, Input, State, dcc, ctx, html
import plotly.graph_objs as go
from RAPDOR.plots import plot_replicate_distribution, plot_distribution, plot_heatmap, plot_barcode_plot, \
    COLOR_SCHEMES
from tempfile import NamedTemporaryFile
from dash_extensions.enrich import callback
from dash.exceptions import PreventUpdate
import logging
import plotly.io as pio

logger = logging.getLogger(__name__)

SVG_RENDERER = pio.renderers["svg"]
SVG_RENDERER.engine = 'kaleido'






FILEEXT = {
    "png": "data:image/png;base64",
    "svg": "data:image/svg+xml;base64"
}







@callback(
    [
        Output("HDBSCAN-cluster-modal", "is_open"),
        Output("DBSCAN-cluster-modal", "is_open"),
        Output("K-Means-cluster-modal", "is_open"),
     ],
    [
        Input("adj-cluster-settings", "n_clicks"),
        Input("HDBSCAN-apply-settings-modal", "n_clicks"),
        Input("DBSCAN-apply-settings-modal", "n_clicks"),
        Input("K-Means-apply-settings-modal", "n_clicks"),

    ],
    [
        State("HDBSCAN-cluster-modal", "is_open"),
        State("DBSCAN-cluster-modal", "is_open"),
        State("K-Means-cluster-modal", "is_open"),
        State("cluster-method", "value")
     ],
    prevent_initial_call=True

)
def _toggle_cluster_modal(n1, n2, n3, n4, hdb_is_open, db_is_open, k_is_open, cluster_method):
    logger.info(f"{ctx.triggered_id} - triggered cluster modal")
    if n1 == 0:
        raise PreventUpdate
    if cluster_method == "HDBSCAN":
        return not hdb_is_open, db_is_open, k_is_open
    elif cluster_method == "DBSCAN":
        return hdb_is_open, not db_is_open, k_is_open
    elif cluster_method == "K-Means":
        return hdb_is_open, db_is_open, not k_is_open
    else:
        return hdb_is_open, db_is_open, k_is_open


# @callback(
#     Output("cluster-img-modal", "is_open"),
#     Output("download-cluster-image", "data"),
#     [
#         Input("cluster-img-modal-btn", "n_clicks"),
#         Input("download-cluster-image-button", "n_clicks"),
#     ],
#     [
#         State("cluster-img-modal", "is_open"),
#         State("cluster-graph", "figure"),
#         State("cluster-download", "value"),
#         State("unique-id", "data"),
#
#     ],
#     prevent_initial_call=True
#
# )
# def _toggle_cluster_image_modal(n1, n2, is_open, graph, filename, uid):
#     logger.info(f"{ctx.triggered_id} - triggered cluster image download modal")
#     if n1 == 0:
#         raise PreventUpdate
#     if ctx.triggered_id == "cluster-img-modal-btn":
#         return not is_open, dash.no_update
#     else:
#         fig = go.Figure(graph)
#         fig.update_layout(
#             font=dict(color="black"),
#             yaxis=dict(gridcolor="black"),
#             xaxis=dict(gridcolor="black"),
#             plot_bgcolor='white',
#
#         )
#         filetype = filename.split(".")[-1]
#         if filetype not in ["svg", "pdf", "png"]:
#             filetype = "svg"
#         with NamedTemporaryFile(suffix=f".{filetype}") as tmpfile:
#             fig.write_image(tmpfile.name, width=1300, height=1300)
#             assert os.path.exists(tmpfile.name)
#             ret_val = dcc.send_file(tmpfile.name)
#             ret_val["filename"] = filename
#         return not is_open, ret_val



