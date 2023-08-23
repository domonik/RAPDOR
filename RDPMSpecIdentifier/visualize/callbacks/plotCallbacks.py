import dash_bootstrap_components as dbc
import numpy as np
from dash import Output, Input, State, ctx, html
from dash.exceptions import PreventUpdate
from plotly import graph_objs as go
from plotly.colors import qualitative
from RDPMSpecIdentifier.plots import plot_replicate_distribution, plot_distribution, plot_barcode_plot, plot_heatmap, \
    plot_dimension_reduction_result
from RDPMSpecIdentifier.visualize.appDefinition import app



COLORS = qualitative.Alphabet + qualitative.Light24

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
    Output("cluster-graph", "figure"),
    Output("alert-div", "children", allow_duplicate=True),
    Input('cluster-feature-slider', 'value'),
    Input("night-mode", "on"),
    Input("primary-open-color-modal", "style"),
    Input("secondary-open-color-modal", "style"),
    Input("recomputation", "children"),
    Input('cluster-method', 'value'),
    Input('dim-red-method', 'value'),
    Input('tbl', 'selected_row_ids'),
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
    prevent_intital_call="initial_duplicate"

)
def update_cluster_graph(
        kernel_size,
        night_mode,
        color,
        color2,
        recomp,
        cluster_method,
        dim_red_method,
        selected_row_ids,
        apply_1,
        apply_2,
        apply_3,
        hdb_min_cluster_size,
        hdb_epsilon,
        db_eps,
        db_min_samples,
        k_clusters,
        k_random_state,
        rdpmsdata
):
    color = color["background-color"], color2["background-color"]
    alert_msg = ""
    try:
        rdpmsdata._calc_cluster_features(kernel_range=kernel_size)
        if cluster_method != "None":
            if cluster_method == "HDBSCAN":
                kwargs = dict(min_cluster_size=hdb_min_cluster_size, cluster_selection_epsilon=hdb_epsilon)
            elif cluster_method == "DBSCAN":
                kwargs = dict(eps=db_eps, min_samples=db_min_samples)
            elif cluster_method == "K-Means":
                kwargs = dict(n_clusters=k_clusters, random_state=k_random_state)
            else:
                raise NotImplementedError("Method Not Implemented")
            clusters = rdpmsdata.cluster_data(method=cluster_method, **kwargs)

        else:
            clusters = None
        embedding = rdpmsdata.reduce_dim(method=dim_red_method)
        colors = COLORS + list(color)
        fig = plot_dimension_reduction_result(
            embedding,
            rdpmsdata,
            name=dim_red_method,
            colors=colors,
            highlight=selected_row_ids,
            clusters=clusters
        )

    except ValueError as error:
        print(str(error))
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
    return fig, alert_msg


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




