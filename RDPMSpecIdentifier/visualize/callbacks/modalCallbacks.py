import os

import dash
import numpy as np
from dash import Output, Input, State, dcc, ctx
import plotly.graph_objs as go
from RDPMSpecIdentifier.plots import plot_replicate_distribution, plot_distribution
from RDPMSpecIdentifier.visualize.appDefinition import app, TMPDIR
import RDPMSpecIdentifier.visualize as rdpv






@app.callback(
    Output("download-image", "data"),
    [
        Input("download-image-button", "n_clicks"),

    ],
    [
        State("named-download", "value"),
        State("protein-id", "children"),
        State("replicate-mode", "on"),
        State("primary-open-color-modal", "style"),
        State("secondary-open-color-modal", "style"),
    ],
    prevent_initial_call=True
)
def _download_image(n_clicks, filename, key, replicate_mode, primary_color, secondary_color):
    key = key.split("Protein ")[-1]
    colors = primary_color['background-color'], secondary_color['background-color']


    filename = os.path.basename(filename)
    array, _ = rdpv.RDPMSDATA[key]
    i = 0
    if rdpv.RDPMSDATA.current_kernel_size is not None:
        i = int(np.floor(rdpv.RDPMSDATA.current_kernel_size / 2))
    if replicate_mode:
        fig = plot_replicate_distribution(array, rdpv.RDPMSDATA.internal_design_matrix, groups="RNAse", offset=i, colors=colors)
    else:
        fig = plot_distribution(array, rdpv.RDPMSDATA.internal_design_matrix, groups="RNAse", offset=i, colors=colors)
    fig.layout.template = "plotly_white"
    fig.update_layout(
        font=dict(color="black"),
        yaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),
        xaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),

    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ))
    fig.update_layout(
        margin={"t": 0, "b": 30, "r": 50},
        font=dict(
            size=16,
        )
    )
    fig.update_xaxes(dtick=1)
    tmpfile = os.path.join(TMPDIR.name, filename)
    fig.write_image(tmpfile)
    assert os.path.exists(tmpfile)
    return dcc.send_file(tmpfile)


@app.callback(
    [
        Output("modal", "is_open"),
        Output("named-download", "value")
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
    filename = key + ".svg"
    if n1 or n2 or n3:
        return not is_open, filename
    return is_open, filename


@app.callback(
    [
        Output("primary-color-modal", "is_open"),
        Output("primary-open-color-modal", "style")
     ],
    [
        Input("primary-open-color-modal", "n_clicks"),
        Input("primary-apply-color-modal", "n_clicks"),
        #Input("select-color", "n_clicks"),
    ],
    [
        State("primary-color-modal", "is_open"),
        State("primary-color-picker", "value")

    ],
    prevent_initial_call=True
)
def _toggle_primary_color_modal(n1, n2, is_open, color_value):
    tid = ctx.triggered_id
    if tid == "primary-open-color-modal":
        return not is_open, dash.no_update
    elif tid == "primary-apply-color-modal":
        rgb = color_value["rgb"]
        r, g, b = rgb["r"], rgb["g"], rgb["b"]
        color = f"rgb({r}, {g}, {b})"
        style = {"width": "100%", "height": "40px", "background-color": color}
    else:
        raise ValueError("")
    return not is_open, style


@app.callback(
    [
        Output("secondary-color-modal", "is_open"),
        Output("secondary-open-color-modal", "style")
     ],
    [
        Input("secondary-open-color-modal", "n_clicks"),
        Input("secondary-apply-color-modal", "n_clicks"),
        #Input("select-color", "n_clicks"),
    ],
    [
        State("secondary-color-modal", "is_open"),
        State("secondary-color-picker", "value")

    ],
    prevent_initial_call=True
)
def _toggle_secondary_color_modal(n1, n2, is_open, color_value):
    tid = ctx.triggered_id
    if tid == "secondary-open-color-modal":
        return not is_open, dash.no_update
    elif tid == "secondary-apply-color-modal":
        rgb = color_value["rgb"]
        r, g, b = rgb["r"], rgb["g"], rgb["b"]
        color = f"rgb({r}, {g}, {b})"
        style = {"width": "100%", "height": "40px", "background-color": color}
    else:
        raise ValueError("")
    return not is_open, style


@app.callback(
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
    if cluster_method == "HDBSCAN":
        return not hdb_is_open, db_is_open, k_is_open
    elif cluster_method == "DBSCAN":
        return hdb_is_open, not db_is_open, k_is_open
    elif cluster_method == "K-Means":
        return hdb_is_open, db_is_open, not k_is_open
    else:
        return hdb_is_open, db_is_open, k_is_open


@app.callback(
    Output("cluster-img-modal", "is_open"),
    Output("download-cluster-image", "data"),
    [
        Input("cluster-img-modal-btn", "n_clicks"),
        Input("download-cluster-image-button", "n_clicks"),
    ],
    [
        State("cluster-img-modal", "is_open"),
        State("cluster-graph", "figure"),
        State("cluster-download", "value"),

    ],
    prevent_initial_call=True

)
def _toggle_cluster_image_modal(n1, n2, is_open, graph, filename):

    if ctx.triggered_id == "cluster-img-modal-btn":
        return not is_open, dash.no_update
    else:
        fig = go.Figure(graph)
        fig.update_layout(
            font=dict(color="black"),
            yaxis=dict(gridcolor="black"),
            xaxis=dict(gridcolor="black"),

        )
        tmpfile = os.path.join(TMPDIR.name, filename)
        fig.write_image(tmpfile)
        assert os.path.exists(tmpfile)
        return not is_open, dcc.send_file(tmpfile)
