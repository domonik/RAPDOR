import dash_bootstrap_components as dbc
import numpy as np
from dash import Output, Input, State, ctx, html
from dash.exceptions import PreventUpdate
from plotly import graph_objs as go

from RDPMSpecIdentifier.plots import plot_replicate_distribution, plot_distribution, plot_barcode_plot, plot_heatmap, \
    plot_dimension_reduction_result
from RDPMSpecIdentifier.visualize.appDefinition import app
import RDPMSpecIdentifier.visualize as rdpv





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

)
def update_distribution_plot(key, kernel_size, primary_color, secondary_color, replicate_mode, night_mode):
    colors = primary_color['background-color'], secondary_color['background-color']
    key = key.split("Protein ")[-1]
    if key is None:
        raise PreventUpdate
    array, _ = rdpv.RDPMSDATA[key]
    i = 0
    if rdpv.RDPMSDATA.current_kernel_size is not None:
        i = int(np.floor(rdpv.RDPMSDATA.current_kernel_size / 2))
    if replicate_mode:
        fig = plot_replicate_distribution(array, rdpv.RDPMSDATA.internal_design_matrix, groups="RNAse", offset=i, colors=colors)
    else:
        fig = plot_distribution(array, rdpv.RDPMSDATA.internal_design_matrix, groups="RNAse", offset=i, colors=colors)
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

)
def update_westernblot(key, kernel_size, primary_color, secondary_color, night_mode):
    colors = primary_color['background-color'], secondary_color['background-color']
    key = key.split("Protein ")[-1]
    if key is None:
        raise PreventUpdate
    array = rdpv.RDPMSDATA.array[rdpv.RDPMSDATA.df.index.get_loc(key)]

    fig = plot_barcode_plot(array, rdpv.RDPMSDATA.internal_design_matrix, groups="RNAse", colors=colors)
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
    State("distance-method", "value")

)
def update_heatmap(key, kernel_size, primary_color, secondary_color, night_mode, distance_method):
    colors = primary_color['background-color'], secondary_color['background-color']
    key = key.split("Protein ")[-1]
    if key is None:
        raise PreventUpdate
    _, distances = rdpv.RDPMSDATA[key]
    fig = plot_heatmap(distances, rdpv.RDPMSDATA.internal_design_matrix, groups="RNAse", colors=colors)
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
    Input('dim-red-btn', 'n_clicks'),
    Input('cluster-feature-slider', 'value'),
    Input("night-mode", "on"),
    Input("primary-open-color-modal", "style"),
    Input("secondary-open-color-modal", "style"),
    Input("recomputation", "children"),
    State('tbl', 'selected_row_ids'),
    prevent_intital_call="initial_duplicate"

)
def update_cluster_graph(clicks, kernel_size, night_mode, color, color2, recomp, selected_row_ids):
    color = color["background-color"], color2["background-color"]
    alert_msg = ""
    try:
        rdpv.RDPMSDATA._calc_cluster_features(kernel_range=kernel_size)
        embedding = rdpv.RDPMSDATA.cluster_shifts()
        fig = plot_dimension_reduction_result(embedding, rdpv.RDPMSDATA, colors=color, highlight=selected_row_ids)

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
        if ctx.triggered_id == "dim-red-btn":
            print("clicks", clicks)
            if clicks != 0:
                alert_msg = html.Div(
                    dbc.Alert(
                        "Get Scores first",
                        color="danger",
                        dismissable=True,
                    ),
                    className="p-2 align-items-center, alert-msg",

                )


    fig.layout.template = "plotly_white"
    if not night_mode:
        fig.update_layout(
            font=dict(color="black"),
            yaxis=dict(gridcolor="black", zeroline=True, color="black"),
            xaxis=dict(gridcolor="black", zeroline=True, color="black"),

        )
    fig.update_layout(
        margin={"t": 0, "b": 30, "r": 50},
        font=dict(
            size=16,
        ),
        xaxis=dict(showline=True, mirror=True, ticks="outside", zeroline=False, ticklen=0, linecolor="black"),
        yaxis=dict(showline=True, mirror=True, ticks="outside", zeroline=False, ticklen=0, linecolor="black")
    )
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    return fig, alert_msg


@app.callback(
    Output("test-div", "children"),
    Input("cluster-graph", "hoverData")
)
def update_plot_with_hover(hover_data):
    if hover_data is None:
        raise PreventUpdate
    hover_data = hover_data["points"][0]
    protein = rdpv.RDPMSDATA.df.loc[hover_data["hovertext"]]["RDPMSpecID"]
    return protein




