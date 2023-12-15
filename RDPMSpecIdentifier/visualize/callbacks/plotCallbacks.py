import dash_bootstrap_components as dbc
import numpy as np
from dash import Output, Input, State, ctx
import dash
from dash.exceptions import PreventUpdate
from plotly import graph_objs as go
from RDPMSpecIdentifier.plots import plot_replicate_distribution, plot_distribution, plot_barcode_plot, plot_heatmap, \
    plot_dimension_reduction, empty_figure, DEFAULT_TEMPLATE, DEFAULT_TEMPLATE_DARK, plot_bars
from dash_extensions.enrich import Serverside, callback
from RDPMSpecIdentifier.datastructures import RDPMSpecData
import logging
import traceback

logger = logging.getLogger(__name__)




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
    if key is None or rdpmsdata is None:
        if key is None:
            fig = empty_figure(
                "No row selected.<br>Click on a row in the table",
                "black" if not night_mode else "white"
            )

        elif rdpmsdata is None:
            fig = empty_figure(
                "There is no data uploaded yet.<br> Please go to the Data upload Page",
                "black" if not night_mode else "white"
            )
    else:
        array = rdpmsdata.norm_array[key]
        i = 0
        if rdpmsdata.state.kernel_size is not None:
            i = int(np.floor(rdpmsdata.state.kernel_size / 2))
        if replicate_mode:
            fig = plot_replicate_distribution(array, rdpmsdata.internal_design_matrix, offset=i, colors=colors)
        else:
            if rdpmsdata.categorical_fraction:
                fig = plot_bars(array, rdpmsdata.internal_design_matrix, x=rdpmsdata.fractions, offset=i,
                                colors=colors)
                if night_mode:
                    fig.update_traces(error_y=dict(color="white"), marker=dict(line=dict(width=1, color="white")))
            else:
                fig = plot_distribution(array, rdpmsdata.internal_design_matrix, offset=i, colors=colors, show_outliers=True)
        if not night_mode:
            fig.layout.template = DEFAULT_TEMPLATE
        else:
            fig.layout.template = DEFAULT_TEMPLATE_DARK

    fig.update_layout(
        margin={"t": 0, "b": 0, "r": 50, "l": 100},
        font=dict(
            size=16,
        ),
        legend=dict(font=dict(size=14)),
        legend2=dict(font=dict(size=14))
    )
    fig.update_xaxes(dtick=1, title=None)
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
        return empty_figure()
    if rdpmsdata is None:
        raise PreventUpdate
    else:
        array = rdpmsdata.array[rdpmsdata.df.index.get_loc(key)]
        fig = plot_barcode_plot(array, rdpmsdata.internal_design_matrix, colors=colors, vspace=0)
        fig.update_yaxes(showticklabels=False, showgrid=False, showline=False)
        fig.update_xaxes(showgrid=False, showticklabels=False, title="", showline=False)
        fig.update_traces(showscale=False)


        fig.update_layout(
            margin={"t": 0, "b": 0, "r": 50, "l": 100},
            font=dict(
                size=16,
            ),
            yaxis=dict(zeroline=False),
            xaxis=dict(zeroline=False),

        )
        fig.update_xaxes(fixedrange=True)
    if not night_mode:
        fig.layout.template = DEFAULT_TEMPLATE
    else:
        fig.layout.template = DEFAULT_TEMPLATE_DARK
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
        distances = rdpmsdata.distances[key]
        fig = plot_heatmap(distances, rdpmsdata.internal_design_matrix, colors=colors)
        fig.update_layout(
            margin={"t": 0, "b": 0, "l": 0, "r": 0}
        )
        fig.update_yaxes(showline=False)
        fig.update_xaxes(showline=False)
    if not night_mode:
        fig.layout.template = DEFAULT_TEMPLATE
    else:
        fig.layout.template = DEFAULT_TEMPLATE_DARK
    return fig, f"Sample {distance_method}"


@callback(
    Output("data-store", "data", allow_duplicate=True),
    Output("plot-dim-red", "data"),
    Input('cluster-method', 'value'),
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
        cluster_method,
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

        if rdpmsdata.cluster_features is None:
            rdpmsdata.calc_distribution_features()
            logger.info("Calculated Cluster Features")
            logger.info("Running Dimension Reduction - because cluster features changed")
        if cluster_method is not None:
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
        return Serverside(rdpmsdata, key=uid), True

    except ValueError as e:
        logger.error(traceback.format_exc())
        return dash.no_update, False



@callback(
    Output("cluster-graph", "figure"),
    Input("night-mode", "on"),
    Input("primary-color", "data"),
    Input("secondary-color", "data"),
    Input("plot-dim-red", "data"),
    Input('current-row-ids', 'data'),
    Input('cluster-marker-slider', 'value'),
    Input('3d-plot', 'on'),

    State('data-store', "data"),
    State("additional-header-dd", "value"),
)
def plot_cluster_results(night_mode, color, color2, plotting, selected_rows, marker_size, td_plot, rdpmsdata: RDPMSpecData, add_header):
    logger.info(f"running cluster plot triggered via - {ctx.triggered_id}")
    dim = 2 if not td_plot else 3
    if dim == 3 and ctx.triggered_id == "cluster-marker-slider":
        raise PreventUpdate
    colors = [color, color2]
    if rdpmsdata is None:
        raise PreventUpdate

    if not plotting:
        fig = empty_figure("Data not Calculated<br> Get Scores first")
    else:
        if selected_rows is not None and len(selected_rows) >= 1:
            highlight = rdpmsdata.df.loc[selected_rows, "RDPMSpecID"]
        else:
            highlight = None
        fig = plot_dimension_reduction(
            rdpmsdata,
            dimensions=dim,
            colors=colors,
            highlight=highlight,
            show_cluster=True if "Cluster" in rdpmsdata.df else False,
            marker_max_size=marker_size,
            second_bg_color="white" if not night_mode else "#181818",
            bubble_legend_color="black" if not night_mode else "white",
            title_col=add_header

        )
    if not night_mode:

        fig.layout.template = DEFAULT_TEMPLATE
    else:
        fig.layout.template = DEFAULT_TEMPLATE_DARK

    fig.update_layout(
        margin={"t": 0, "b": 30, "r": 50},
        font=dict(
            size=16,
        ),
        xaxis2=dict(showline=True, mirror=True, ticks="outside", zeroline=False, ticklen=0, linecolor="black"),
        yaxis2=dict(showline=True, mirror=True, ticks="outside", zeroline=False, ticklen=0, linecolor="black"),
        plot_bgcolor='#222023',

    )
    if not night_mode:
        fig.update_layout(
            font=dict(color="black"),
            yaxis2=dict(gridcolor="black", zeroline=False, color="black", linecolor="black"),
            xaxis2=dict(gridcolor="black", zeroline=False, color="black", linecolor="black"),
            plot_bgcolor='#e1e1e1',


        )
        if plotting and dim == 3:
            fig.update_scenes(
                xaxis_backgroundcolor="#e1e1e1",
                yaxis_backgroundcolor="#e1e1e1",
                zaxis_backgroundcolor="#e1e1e1",
            )
    else:
        if plotting and dim == 3:
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




