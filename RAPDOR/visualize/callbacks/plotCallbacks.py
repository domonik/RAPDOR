import dash_bootstrap_components as dbc
import numpy as np
from dash import Output, Input, State, ctx
import dash
from dash.exceptions import PreventUpdate
from plotly import graph_objs as go
from RAPDOR.plots import plot_replicate_distribution, plot_distribution, plot_barcode_plot, plot_heatmap, \
    plot_dimension_reduction, empty_figure, DEFAULT_TEMPLATE, DEFAULT_TEMPLATE_DARK, plot_bars, plot_distance_and_var
from dash_extensions.enrich import Serverside, callback
from RAPDOR.datastructures import RAPDORData
import logging
import traceback
from pandas.api.types import is_numeric_dtype
logger = logging.getLogger(__name__)

@callback(
    Output("distribution-graph", "figure"),
    [
        Input("current-protein-id", "data"),
        Input('recomputation', 'children'),
        Input("primary-color", "data"),
        Input("secondary-color", "data"),
        Input("replicate-mode", "on"),
        Input("night-mode", "on"),
        Input("raw-plot", "on")
    ],
    State("data-store", "data"),
    prevent_initial_call=True

)
def update_distribution_plot(key, recomp, primary_color, secondary_color, replicate_mode, night_mode, raw, rapdordata):
    logger.info(f"{ctx.triggered_id} triggered update of distribution plot")
    colors = primary_color, secondary_color
    if key is None or rapdordata is None:
        if key is None:
            fig = empty_figure(
                "No row selected.<br>Click on a row in the table",
                "black" if not night_mode else "white"
            )

        elif rapdordata is None:
            fig = empty_figure(
                "There is no data uploaded yet.<br> Please go to the Data upload Page",
                "black" if not night_mode else "white"
            )
    else:
        array = rapdordata.norm_array[key] if not raw else rapdordata.kernel_array[key]
        mode = "raw" if raw else "rel."
        yname = f"{mode} {rapdordata.measure_type} {rapdordata.measure}"
        i = 0
        if rapdordata.state.kernel_size is not None:
            i = int(np.floor(rapdordata.state.kernel_size / 2))
        if replicate_mode:
            fig = plot_replicate_distribution(array, rapdordata.internal_design_matrix, offset=i, colors=colors, yname=yname)
        else:
            if rapdordata.categorical_fraction:
                fig = plot_bars(array, rapdordata.internal_design_matrix, x=rapdordata.fractions, offset=i,
                                colors=colors, yname=yname)
                fig.update_traces(error_y=dict(width=15))
            else:
                fig = plot_distribution(array, rapdordata.internal_design_matrix, offset=i, colors=colors, show_outliers=True, yname=yname)
        if not night_mode:
            fig.layout.template = DEFAULT_TEMPLATE
        else:
            fig.layout.template = DEFAULT_TEMPLATE_DARK

        q_x = fig.layout.legend.x - 0.02
        q_y = (fig.layout.legend2.y - fig.layout.legend.y) / 2 + fig.layout.legend.y
        fig.add_annotation(
            text=u"\u2B24",
            hovertext="Click on legend traces to de-/activate them",
            x=q_x,
            y=q_y + 0.01,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            #bgcolor="white" if night_mode else "grey",
            font=dict(color="white" if night_mode else "black", size=24, family="Font Awesome 6 Free"),

        )
        fig.add_annotation(
            text="<b>?</b>",
            hovertext="Click on legend traces to de-/activate them",
            x=q_x,
            y=q_y,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            #bgcolor="white" if night_mode else "grey",
            font=dict(color="black" if night_mode else "white", size=20, family="Font Awesome 6 Free"),

        )



    fig.update_layout(
        margin={"t": 0, "b": 0, "r": 50, "l": 100},
        font=dict(
            size=16,
        ),
        legend=dict(font=dict(size=14)),
        legend2=dict(font=dict(size=14))
    )
    if isinstance(rapdordata.fractions[0], str):
        fig.update_xaxes(
            tickvals=list(range(len(rapdordata.fractions))),
            ticktext=[val.replace(" ", "<br>").replace("<br>&<br>", " &<br>") for val in rapdordata.fractions],
            tickmode="array"
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
def update_westernblot(key, kernel_size, primary_color, secondary_color, night_mode, rapdordata):
    colors = primary_color, secondary_color
    if key is None:
        return empty_figure()
    if rapdordata is None:
        raise PreventUpdate
    else:
        array = rapdordata.array[rapdordata.df.index.get_loc(key)]
        fig = plot_barcode_plot(array, rapdordata.internal_design_matrix, colors=colors, vspace=0)
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
def update_heatmap(key, recomp, primary_color, secondary_color, night_mode, distance_method, rapdordata):
    colors = primary_color, secondary_color
    if key is None:
        fig = empty_figure(
                "No row selected.<br>Click on a row in the table",
                "black" if not night_mode else "white"
            )
    else:
        if rapdordata is None:
            raise PreventUpdate
        else:
            distances = rapdordata.distances[key]
            fig = plot_heatmap(distances, rapdordata.internal_design_matrix, colors=colors)
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
        rapdordata: RAPDORData,
        uid
):
    logger.info(f"{ctx.triggered_id} - triggered cluster-callback")
    if rapdordata is None:
        raise PreventUpdate
    try:

        if rapdordata.cluster_features is None:
            rapdordata.calc_distribution_features()
            logger.info("Calculated Cluster Features")
            logger.info("Running Bubble Plot - because cluster features changed")
        if cluster_method is not None:
            if cluster_method == "HDBSCAN":
                kwargs = dict(min_cluster_size=hdb_min_cluster_size, cluster_selection_epsilon=hdb_epsilon)
            elif cluster_method == "DBSCAN":
                kwargs = dict(eps=db_eps, min_samples=db_min_samples)
            elif cluster_method == "K-Means":
                kwargs = dict(n_clusters=k_clusters, random_state=k_random_state)
            else:
                raise NotImplementedError("Method Not Implemented")
            if rapdordata.state.cluster_method != cluster_method or rapdordata.state.cluster_args != kwargs:
                logger.info("Running Clustering")
                clusters = rapdordata.cluster_data(method=cluster_method, **kwargs, )
        else:
            rapdordata.remove_clusters()
        return Serverside(rapdordata, key=uid), True

    except ValueError as e:
        logger.info(traceback.format_exc())
        return dash.no_update, False



@callback(
    Output("cutoff-type", "value"),
    Output("cutoff-type", "options"),
    Output("plot-cutoff-name", "children"),
    Output("cutoff-type", "clearable"),
    Input("data-store", "data"),
    Input("dataset-plot-type", "value"),
    State("cutoff-type", "value"),

)
def update_cutoff_selection(rapdordata: RAPDORData, plot_type, current_selection):
    if rapdordata is None:
        raise PreventUpdate
    options = [option for option in rapdordata.score_columns if
               option in rapdordata.df and is_numeric_dtype(rapdordata.df[option])]
    selection = dash.no_update

    if plot_type == "Bubble Plot":
        name = "Cutoff Type"
        clearable = True
    else:
        name = "Y Axis"
        selection = "ANOSIM R" if len(options) > 0 and current_selection not in options else selection
        clearable = False
        options = [option for option in options if ("p-Value" in option) or option in ("ANOSIM R", "PERMANOVA F")]
    if len(options) > 0 and current_selection not in options:
        selection = None
    elif len(options) == 0:
        options = dash.no_update
    return selection, options, name, clearable

@callback(
    Output("cutoff-range", "min"),
    Output("cutoff-range", "max"),
    Output("cutoff-range", "marks"),
    Output("cutoff-range", "value"),
    Output("cutoff-range", "disabled"),
    Input("cutoff-type", "value"),
    Input("dataset-plot-type", "value"),
    State("data-store", "data")

)
def update_range_slider(cutoff_type, plot_type, rapdordata: RAPDORData):
    if plot_type != "Bubble Plot":
        marks = [0, 1]
        marks_t = {i: f"" for i in marks}
        marks = []
        return 0, 1, marks_t, marks, True
    if cutoff_type is not None:
        min_v = np.nanmin(rapdordata.df[cutoff_type])
        max_v = np.nanmax(rapdordata.df[cutoff_type])
        min_v = np.floor(min_v * 100) / 100
        max_v = np.ceil(max_v * 100) / 100
        if "p-Value" in cutoff_type:
            min_v = min(1e-5, min_v)
            if min_v == 0:
                min_v = 1e-20
            max_v = 1
            min_v = np.floor(np.log10(min_v))
            max_v = np.ceil(np.log10(max_v))
            marks = np.linspace(min_v, max_v, 5)
            marks = np.floor(marks)
            p = np.log10(0.05)
            iidx = np.searchsorted(marks, p)
            marks = np.insert(marks, iidx, p)
            marks_t = {int(i) if iidx != idx else i: dict(label=f"{(10**i):.0e}", style={"color": "var(--secondary-color)"} if idx == iidx else {"color": "r-text-color"}) for idx, i in enumerate(marks)}
            min_v = marks[0]
            max_v = marks[-1]
            d_max = marks[iidx]


        else:
            marks = np.linspace(min_v, max_v, 5)
            d_max = marks[-1]
            marks_t = {i: f"{i:.1f}" for i in marks}
        logger.info(f"updating marks {marks_t} -min {min_v} -max {max_v}")
        disabled = False
    else:
        min_v = None
        max_v = None
        marks_t = None
        d_max = None
        disabled = True
    logger.info(f"updating range slider: {min_v, max_v, marks_t, d_max, disabled}")
    return min_v, max_v, marks_t, (min_v, d_max), disabled




@callback(
    Output("showLFC", "on"),
    Output("showLFC", "disabled"),
    Output("3d-plot", "disabled"),
    Input("3d-plot", "on"),
    Input("dataset-plot-type", "value"),
)
def disable_lfc_and_3d(tdplot, plot_type):
    if plot_type == "Bubble Plot":
        if tdplot:
            return False, True, False
        else:
            return dash.no_update, False, False
    else:
        return dash.no_update, False, True





@callback(
    Output("cluster-graph", "figure"),
    Input("night-mode", "on"),
    Input("dataset-plot-type", "value"),
    Input("primary-color", "data"),
    Input("secondary-color", "data"),
    Input("plot-dim-red", "data"),
    Input('current-row-ids', 'data'),
    Input('cluster-marker-slider', 'value'),
    Input('3d-plot', 'on'),
    Input('cutoff-range', 'value'),
    Input("additional-header-dd", "value"),
    Input("showLFC", "on"),
    State('cutoff-type', 'value'),
    State('data-store', "data"),
)
def plot_cluster_results(night_mode, plot_type, color, color2, plotting, selected_rows, marker_size, td_plot, cutoff_range, add_header, show_lfc, cutoff_type, rapdordata: RAPDORData):
    logger.info(f"running cluster plot triggered via - {ctx.triggered_id}")
    if rapdordata is None:
        raise PreventUpdate
    colors = [color, color2]
    dim = 2 if not td_plot else 3
    if not plotting:
        fig = empty_figure("Data not Calculated<br> Get Scores first")
    else:
        if selected_rows is not None and len(selected_rows) >= 1:
            highlight = rapdordata.df.loc[selected_rows, "RAPDORid"]
        else:
            highlight = None
        if plot_type == "Bubble Plot":
            if dim == 3 and ctx.triggered_id == "cluster-marker-slider":
                raise PreventUpdate

            logger.info(f"Cutoff - {cutoff_range}")
            logger.info(f"highlight - {highlight}")
            if cutoff_type is None:
                cutoff_range = None
            else:
                if "p-Value" in cutoff_type:
                    cutoff_range = 10 ** cutoff_range[0], 10 ** cutoff_range[1]
            if show_lfc:
                highlight_color = "white" if night_mode else "black"
            else:
                highlight_color = None
            fig = plot_dimension_reduction(
                rapdordata,
                dimensions=dim,
                colors=colors,
                highlight=highlight,
                show_cluster=True if "Cluster" in rapdordata.df else False,
                marker_max_size=marker_size,
                second_bg_color="white" if not night_mode else "#181818",
                bubble_legend_color="black" if not night_mode else "white",
                title_col=add_header,
                cutoff_range=cutoff_range,
                cutoff_type=cutoff_type,
                highlight_color=highlight_color,
                show_lfc=show_lfc

            )
        else:
            if cutoff_type is None:
                fig = empty_figure("Select Y Axis Type first")
            else:
                fig = plot_distance_and_var(rapdordata, colors, title_col=add_header, highlight=highlight, var_type=cutoff_type, show_lfc=show_lfc)
                fig.update_traces(marker=dict(size=marker_size))
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




