import dash_bootstrap_components as dbc
import numpy as np
from dash import Output, Input, State, ctx
import dash
from dash.exceptions import PreventUpdate
from plotly import graph_objs as go
from plotly.colors import qualitative
from RDPMSpecIdentifier.plots import plot_replicate_distribution, plot_distribution, plot_barcode_plot, plot_heatmap, \
    plot_dimension_reduction_result, empty_figure
from dash_extensions.enrich import Serverside, callback
from RDPMSpecIdentifier.datastructures import RDPMSpecData
import logging

logger = logging.getLogger(__name__)

COLORS = qualitative.Alphabet + qualitative.Light24 + qualitative.Dark24 + qualitative.G10




@callback(
    Output("distribution-graph", "figure"),
    [
        Input("current-protein-id", "data"),
        Input('recomputation', 'children'),
        Input("primary-color", "data"),
        Input("secondary-color", "data"),
        Input("replicate-mode", "on"),
        Input("night-mode", "on")
    ],
    State("data-store", "data"),
    prevent_initial_call=True

)
def update_distribution_plot(key, kernel_size, primary_color, secondary_color, replicate_mode, night_mode, rdpmsdata):
    logger.info(f"{ctx.triggered_id} triggered update of distribution plot")
    colors = primary_color, secondary_color
    if key is None:
        raise PreventUpdate
    if rdpmsdata is None:
        fig = empty_figure(
            "There is no data uploaded yet.<br> Please go to the Data upload Page",
            "black" if not night_mode else "white"
        )

    else:
        array, _ = rdpmsdata[key]
        i = 0
        if rdpmsdata.state.kernel_size is not None:
            i = int(np.floor(rdpmsdata.state.kernel_size / 2))
        if replicate_mode:
            fig = plot_replicate_distribution(array, rdpmsdata.internal_design_matrix, groups="RNase", offset=i, colors=colors)
        else:
            fig = plot_distribution(array, rdpmsdata.internal_design_matrix, groups="RNase", offset=i, colors=colors)
        fig.layout.template = "plotly_white"
        if not night_mode:
            fig.update_layout(
                font=dict(color="black"),
                yaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),
                xaxis=dict(gridcolor="black", zeroline=True, zerolinecolor="black"),

            )
    fig.update_layout(
        margin={"t": 0, "b": 30, "r": 50, "l": 100},
        font=dict(
            size=16,
        )
    )
    fig.update_xaxes(dtick=1)
    fig.update_xaxes(fixedrange=True)
    return fig


@callback(
    Output("westernblot-graph", "figure"),
    [
        Input("current-protein-id", "data"),
        Input('recomputation', 'children'),
        Input("primary-color", "data"),
        Input("secondary-color", "data"),
        Input("night-mode", "on"),
    ],
    State("data-store", "data")

)
def update_westernblot(key, kernel_size, primary_color, secondary_color, night_mode, rdpmsdata):
    colors = primary_color, secondary_color
    if key is None:
        raise PreventUpdate
    if rdpmsdata is None:
        raise PreventUpdate
    else:
        array = rdpmsdata.array[rdpmsdata.df.index.get_loc(key)]
        fig = plot_barcode_plot(array, rdpmsdata.internal_design_matrix, groups="RNase", colors=colors)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        fig.update_xaxes(showgrid=False, showticklabels=False)
        if not night_mode:
            fig.update_layout(
                font=dict(color="black"),

            )
        fig.update_layout(
            margin={"t": 0, "b": 0, "r": 50, "l": 100},
            font=dict(
                size=16,
            ),
            yaxis=dict(zeroline=False),
            xaxis=dict(zeroline=False),

        )
        fig.update_xaxes(fixedrange=True)

        fig.layout.template = "plotly_white"
    return fig


@callback(
    [
        Output("heatmap-graph", "figure"),
        Output("distance-header", "children")
    ],
    [
        Input("current-protein-id", "data"),
        Input('recomputation', 'children'),
        Input("primary-color", "data"),
        Input("secondary-color", "data"),
        Input("night-mode", "on"),

    ],
    State("distance-method", "value"),
    State("data-store", "data")

)
def update_heatmap(key, recomp, primary_color, secondary_color, night_mode, distance_method, rdpmsdata):
    colors = primary_color, secondary_color
    if key is None:
        raise PreventUpdate
    if rdpmsdata is None:
        raise PreventUpdate
    else:
        _, distances = rdpmsdata[key]
        fig = plot_heatmap(distances, rdpmsdata.internal_design_matrix, groups="RNase", colors=colors)
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


@callback(
    Output("data-store", "data", allow_duplicate=True),
    Output("plot-dim-red", "data"),
    Input('cluster-feature-slider', 'value'),
    Input('3d-plot', 'on'),
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
        td_plot,
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
    logger.info(f"{ctx.triggered_id} - triggered cluster-callback")
    if rdpmsdata is None:
        raise PreventUpdate
    try:
        dim = 2 if not td_plot else 3

        if ctx.triggered_id == "cluster-feature-slider" or rdpmsdata.cluster_features is None:
            if rdpmsdata.state.cluster_kernel_distance != kernel_size:
                rdpmsdata.calc_cluster_features(kernel_range=kernel_size)
                logger.info("Calculated Cluster Features")

                rdpmsdata.set_embedding(dim, method=reduction_method)
                logger.info("Running Dimension Reduction - because cluster features changed")
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
                if rdpmsdata.state.cluster_method != cluster_method or rdpmsdata.state.cluster_args != kwargs:
                    logger.info("Running Clustering")
                    clusters = rdpmsdata.cluster_data(method=cluster_method, **kwargs, )
            else:
                rdpmsdata.remove_clusters()
        if ctx.triggered_id == "dim-red-method" or rdpmsdata.current_embedding is None or ctx.triggered_id == "3d-plot":
            if rdpmsdata.current_embedding is None or rdpmsdata.current_embedding.shape[-1] != dim or rdpmsdata.state.dimension_reduction != reduction_method:
                logger.info("Running Dimension Reduction")
                rdpmsdata.set_embedding(dim, method=reduction_method)


        return Serverside(rdpmsdata, key=uid), True

    except ValueError:
        return dash.no_update, False



@callback(
    Output("cluster-graph", "figure"),
    Input("night-mode", "on"),
    Input("primary-color", "data"),
    Input("secondary-color", "data"),
    Input("plot-dim-red", "data"),
    Input('tbl', 'selected_row_ids'),
    State('data-store', "data"),
)
def plot_cluster_results(night_mode, color, color2, plotting, selected_rows, rdpmsdata: RDPMSpecData):

    color = color, color2
    colors = COLORS + list(color)
    if rdpmsdata is None:
        raise PreventUpdate

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
            name=rdpmsdata.state.dimension_reduction,
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
        if plotting and rdpmsdata.current_embedding.shape[-1] == 3:
            fig.update_scenes(
                xaxis_backgroundcolor="#e1e1e1",
                yaxis_backgroundcolor="#e1e1e1",
                zaxis_backgroundcolor="#e1e1e1",
            )
    else:
        if plotting and rdpmsdata.current_embedding.shape[-1] == 3:
            fig.update_scenes(
                xaxis_backgroundcolor="#222023",
                yaxis_backgroundcolor="#222023",
                zaxis_backgroundcolor="#222023",
            )
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    return fig



@callback(
    Output("test-div", "children"),
    Input("cluster-graph", "hoverData"),
    Input("cluster-graph", "clickData"),
)
def update_plot_with_hover(hover_data, click_data):
    logger.info("Hover Callback triggered")
    if hover_data is None and click_data is None:
        raise PreventUpdate
    else:
        logger.info(ctx.triggered_prop_ids)
        if "cluster-graph.hoverData" in ctx.triggered_prop_ids:
            hover_data = hover_data["points"][0]
        else:
            hover_data = click_data["points"][0]

        split_l = hover_data["hovertext"].split(": ")
        p_id, protein = split_l[0], split_l[1]
    return p_id




