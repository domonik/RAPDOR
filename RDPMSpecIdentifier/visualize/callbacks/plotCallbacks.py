import dash_bootstrap_components as dbc
import numpy as np
from dash import Output, Input, State, ctx, html
import dash
from dash.exceptions import PreventUpdate
from plotly import graph_objs as go
from plotly.colors import qualitative
from RDPMSpecIdentifier.plots import plot_replicate_distribution, plot_distribution, plot_barcode_plot, plot_heatmap, \
    plot_dimension_reduction_result
from RDPMSpecIdentifier.visualize.appDefinition import app
from dash_extensions.enrich import Serverside
from RDPMSpecIdentifier.datastructures import RDPMSpecData


COLORS = qualitative.Alphabet + qualitative.Light24 + qualitative.Dark24 + qualitative.G10

@app.callback(
    Output("distribution-graph", "figure"),
    [
        Input("protein-id", "children"),
        Input('recomputation', 'children'),
        Input("primary-open-color-modal", "style"),
        Input("secondary-open-color-modal", "style"),
        Input("replicate-mode", "on"),
        Input("night-mode", "on")
    ],
    State("data-store", "data")

)
def update_distribution_plot(key, kernel_size, primary_color, secondary_color, replicate_mode, night_mode, rdpmsdata):
    colors = primary_color['background-color'], secondary_color['background-color']
    key = key.split("Protein ")[-1]
    if key is None:
        raise PreventUpdate
    array, _ = rdpmsdata[key]
    i = 0
    if rdpmsdata.current_kernel_size is not None:
        i = int(np.floor(rdpmsdata.current_kernel_size / 2))
    if replicate_mode:
        fig = plot_replicate_distribution(array, rdpmsdata.internal_design_matrix, groups="RNAse", offset=i, colors=colors)
    else:
        fig = plot_distribution(array, rdpmsdata.internal_design_matrix, groups="RNAse", offset=i, colors=colors)
    fig.layout.template = "plotly_white"
    if not night_mode:
        fig.update_layout(
            font=dict(color="black"),
            yaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),
            xaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),

        )
    fig.update_layout(
        margin={"t": 0, "b": 30, "r": 50},
        font=dict(
            size=16,
        )
    )
    fig.update_xaxes(dtick=1)
    fig.update_xaxes(fixedrange=True)
    return fig


@app.callback(
    Output("westernblot-graph", "figure"),
    [
        Input("protein-id", "children"),
        Input('recomputation', 'children'),
        Input("primary-open-color-modal", "style"),
        Input("secondary-open-color-modal", "style"),
        Input("night-mode", "on"),
    ],
    State("data-store", "data")

)
def update_westernblot(key, kernel_size, primary_color, secondary_color, night_mode, rdpmsdata):
    colors = primary_color['background-color'], secondary_color['background-color']
    key = key.split("Protein ")[-1]
    if key is None:
        raise PreventUpdate
    array = rdpmsdata.array[rdpmsdata.df.index.get_loc(key)]

    fig = plot_barcode_plot(array, rdpmsdata.internal_design_matrix, groups="RNAse", colors=colors)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_xaxes(showgrid=False, showticklabels=False)
    if not night_mode:
        fig.update_layout(
            font=dict(color="black"),
            yaxis=dict(gridcolor="black"),
            xaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),

        )
    fig.update_layout(
        margin={"t": 0, "b": 0, "r": 50},
        font=dict(
            size=16,
        )
    )
    fig.update_xaxes(fixedrange=True)

    fig.layout.template = "plotly_white"
    return fig


@app.callback(
    [
        Output("heatmap-graph", "figure"),
        Output("distance-header", "children")
    ],
    [
        Input("protein-id", "children"),
        Input('recomputation', 'children'),
        Input("primary-open-color-modal", "style"),
        Input("secondary-open-color-modal", "style"),
        Input("night-mode", "on"),

    ],
    State("distance-method", "value"),
    State("data-store", "data")

)
def update_heatmap(key, kernel_size, primary_color, secondary_color, night_mode, distance_method, rdpmsdata):
    colors = primary_color['background-color'], secondary_color['background-color']
    key = key.split("Protein ")[-1]
    if key is None:
        raise PreventUpdate
    _, distances = rdpmsdata[key]
    fig = plot_heatmap(distances, rdpmsdata.internal_design_matrix, groups="RNAse", colors=colors)
    fig.layout.template = "plotly_white"
    if not night_mode:
        fig.update_layout(
            font=dict(color="black"),
            yaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),
            xaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),

        )
    fig.update_layout(
        margin={"t": 0, "b": 0, "l": 0, "r": 0}
    )
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    return fig, f"Sample {distance_method}"


@app.callback(
    Output("data-store", "data", allow_duplicate=True),
    Output("plot-dim-red", "data"),
    Input('cluster-feature-slider', 'value'),
    Input('cluster-method', 'value'),
    Input('dim-red-method', 'value'),
    Input("recomputation", "children"),
    Input("run-clustering", "data"),
    Input("HDBSCAN-apply-settings-modal", "n_clicks"),
    Input("DBSCAN-apply-settings-modal", "n_clicks"),
    Input("K-Means-apply-settings-modal", "n_clicks"),
    State('HDBSCAN-min_cluster_size-input', "value"),
    State('HDBSCAN-cluster_selection_epsilon-input', "value"),
    State('DBSCAN-eps-input', "value"),
    State('DBSCAN-min_samples-input', "value"),
    State('K-Means-n_clusters-input', "value"),
    State('K-Means-random_state-input', "value"),
    State('data-store', "data"),
    State('unique-id', "data"),
    prevent_intital_call="initial_duplicate"
)
def calc_clusters(
        kernel_size,
        cluster_method,
        reduction_method,
        recomp,
        run_cluster,
        apply_1,
        apply_2,
        apply_3,
        hdb_min_cluster_size,
        hdb_epsilon,
        db_eps,
        db_min_samples,
        k_clusters,
        k_random_state,
        rdpmsdata: RDPMSpecData,
        uid
):
    print(ctx.triggered_id)
    try:
        if ctx.triggered_id == "cluster-feature-slider" or rdpmsdata.cluster_features is None:
            rdpmsdata._calc_cluster_features(kernel_range=kernel_size)
        if ctx.triggered_id != "dim-red-method":
            if cluster_method != "None":
                if cluster_method == "HDBSCAN":
                    kwargs = dict(min_cluster_size=hdb_min_cluster_size, cluster_selection_epsilon=hdb_epsilon)
                elif cluster_method == "DBSCAN":
                    kwargs = dict(eps=db_eps, min_samples=db_min_samples)
                elif cluster_method == "K-Means":
                    kwargs = dict(n_clusters=k_clusters, random_state=k_random_state)
                else:
                    raise NotImplementedError("Method Not Implemented")
                clusters = rdpmsdata.cluster_data(method=cluster_method, **kwargs, )
            else:
                rdpmsdata.remove_clusters()
        if ctx.triggered_id == "dim-red-method" or rdpmsdata.current_embedding is None or ctx.triggered_id == "cluster-feature-slider":
            rdpmsdata.set_embedding(2, method=reduction_method)

        return Serverside(rdpmsdata, key=uid), True

    except ValueError:
        return dash.no_update, False



@app.callback(
    Output("cluster-graph", "figure"),
    Input("night-mode", "on"),
    Input("primary-open-color-modal", "style"),
    Input("secondary-open-color-modal", "style"),
    Input("plot-dim-red", "data"),
    Input('tbl', 'selected_row_ids'),
    State('data-store', "data"),
)
def plot_cluster_results(night_mode, color, color2, plotting, selected_rows, rdpmsdata: RDPMSpecData):

    color = color["background-color"], color2["background-color"]
    colors = COLORS + list(color)

    if not plotting:
        fig = go.Figure()
        fig.add_annotation(
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="middle",
            x=0.5,
            y=0.5,
            text="Data not Calculated<br> Get Scores first",
            showarrow=False,
            font=(dict(size=28))
        )
    else:
        fig = plot_dimension_reduction_result(
            rdpmsdata.current_embedding,
            rdpmsdata,
            name=rdpmsdata.current_dim_red_method,
            colors=colors,
            highlight=selected_rows,
            clusters=rdpmsdata.df["Cluster"] if "Cluster" in rdpmsdata.df else None
        )
    fig.layout.template = "plotly_white"

    fig.update_layout(
        margin={"t": 30, "b": 30, "r": 50},
        font=dict(
            size=16,
        ),
        xaxis=dict(showline=True, mirror=True, ticks="outside", zeroline=False, ticklen=0, linecolor="black"),
        yaxis=dict(showline=True, mirror=True, ticks="outside", zeroline=False, ticklen=0, linecolor="black"),
        plot_bgcolor='#222023',

    )
    if not night_mode:
        fig.update_layout(
            font=dict(color="black"),
            yaxis=dict(gridcolor="black", zeroline=False, color="black", linecolor="black"),
            xaxis=dict(gridcolor="black", zeroline=False, color="black", linecolor="black"),
            plot_bgcolor='#e1e1e1',


        )
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    return fig



@app.callback(
    Output("test-div", "children"),
    Input("cluster-graph", "hoverData"),
    State("data-store", "data")
)
def update_plot_with_hover(hover_data, rdpmsdata):
    if hover_data is None:
        raise PreventUpdate
    hover_data = hover_data["points"][0]
    protein = rdpmsdata.df.loc[hover_data["hovertext"]]["RDPMSpecID"]
    return protein




