import os

import dash
import numpy as np
from dash import Output, Input, State, dcc, ctx, html
import plotly.graph_objs as go
from RDPMSpecIdentifier.plots import plot_replicate_distribution, plot_distribution, plot_heatmap, plot_barcode_plot
from tempfile import NamedTemporaryFile
from dash_extensions.enrich import callback
from dash.exceptions import PreventUpdate
import logging
import plotly.io as pio
from RDPMSpecIdentifier.visualize.staticContent import COLOR_SCHEMES

logger = logging.getLogger(__name__)

SVG_RENDERER = pio.renderers["svg"]
SVG_RENDERER.engine = 'kaleido'




@callback(
    [
        Output("modal", "is_open"),
        Output("download-name-input", "value")
     ],
    [
        Input("open-modal", "n_clicks"),
        Input("close", "n_clicks"),
        Input("download-image-button", "n_clicks"),
    ],
    [State("modal", "is_open"),
     State("protein-id", "children")
     ],
    prevent_initial_call=True

)
def _toggle_modal(n1, n2, n3, is_open, key):
    key = key.split("Protein ")[-1]
    logger.info(f"{ctx.triggered_id} - triggered download modal")
    filename = key + ".svg"
    if ctx.triggered_id in ["open-modal", "close"]:
        return not is_open, filename
    elif n3:
        logger.info("Closing Modal because of Download")
        return not is_open, dash.no_update
    return is_open, filename


@callback(
    [
        Output("distribution-graph-download-preview", "children"),
        Output("download-image", "data"),

    ],
    [
        Input("modal", "is_open"),
        Input("plot-type-radio", "value"),
        Input("download-image-button", "n_clicks"),
        Input("download-width-input", "value"),
        Input("download-height-input", "value"),
        Input("download-marker-size-input", "value"),
        Input("download-line-width-input", "value"),
        Input("download-grid-width-input", "value")

    ],
    [
        State("download-name-input", "value"),
        State("current-protein-id", "data"),
        State("replicate-mode", "on"),
        State("primary-color", "data"),
        State("secondary-color", "data"),
        State("data-store", "data"),
        State("unique-id", "data"),
        State("distance-method", "value"),

    ],
    prevent_initial_call=True

)
def update_image_preview(is_open, plot_type, dwnld, img_width, img_height, marker_size, line_width, grid_width, filename, key, replicate_mode, primary_color, secondary_color, rdpmsdata, uid, distance_method):
    logger.info(f"{ctx.triggered_id} - triggered image download modal changes")
    try:
        img_width = max(min(img_width, 2000), 100)
        img_height = max(min(img_height, 2000), 100)
    except TypeError:
        img_height = 500
        img_width = 800
    send_data = dash.no_update
    if not is_open and ctx.triggered_id != "download-image-button":
        raise PreventUpdate
    colors = primary_color, secondary_color
    array, _ = rdpmsdata[key]
    i = 0
    if rdpmsdata.state.kernel_size is not None:
        i = int(np.floor(rdpmsdata.state.kernel_size / 2))
    if plot_type == 0:
        if replicate_mode:
            fig = plot_replicate_distribution(array, rdpmsdata.internal_design_matrix, groups="RNase", offset=i,
                                              colors=colors)
        else:
            fig = plot_distribution(array, rdpmsdata.internal_design_matrix, groups="RNase", offset=i, colors=colors)
        fig.update_layout(
            yaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black", gridwidth=grid_width),
            xaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black", gridwidth=grid_width),

        )
        fig.update_traces(
            marker=dict(size=marker_size,
                        ),
            line=dict(
                width=line_width
            )
        )
    elif plot_type == 1:
        _, distances = rdpmsdata[key]
        fig = plot_heatmap(distances, rdpmsdata.internal_design_matrix, groups="RNase", colors=colors)
        title = dict(
            text=f"Sample {distance_method}",
            font=dict(size=24),
            automargin=True,
            x=0.5,
            xref="paper",
            xanchor="center"
        )
        fig.update_layout(title=title)
        fig.update_yaxes(linewidth=grid_width)
        fig.update_xaxes(linewidth=grid_width)
    elif plot_type == 2:
        fig = plot_barcode_plot(array, rdpmsdata.internal_design_matrix, groups="RNase", colors=colors)
        fig.update_yaxes(linewidth=grid_width)
        fig.update_xaxes(linewidth=grid_width)

    else:
        raise PreventUpdate
    fig.layout.template = "plotly_white"
    fig.update_layout(
        margin={"t": 0, "b": 0, "r": 0},
        font=dict(
            size=16,
            color="black"
        ),
        width=img_width,
        height=img_height
    )
    fig.update_xaxes(dtick=1)
    filetype = filename.split(".")[-1]
    if filetype not in ["svg", "png"]:
        filetype = "svg"

    with NamedTemporaryFile(suffix=f".{filetype}") as tmpfile:
        fig.write_image(tmpfile.name)
        assert os.path.exists(tmpfile.name)
        ret_val = dcc.send_file(tmpfile.name)
        ret_val["filename"] = filename
        encoded_image = ret_val["content"]
        fig = html.Img(
            src=f'{FILEEXT[filetype]},{encoded_image}',
            style={"margin-left": "auto", "margin-right": "auto", "display": "block"}
        )
    if ctx.triggered_id == "download-image-button":
        logger.info("Downloading Image")
        fig = dash.no_update
        send_data = ret_val
    return fig, send_data

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



