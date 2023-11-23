
import dash
from dash import html
import base64
from dash import dcc
from dash_extensions.enrich import callback, Input, Output, Serverside, State
from RDPMSpecIdentifier.visualize import DISPLAY
from RDPMSpecIdentifier.datastructures import RDPMSpecData
import logging
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from RDPMSpecIdentifier.plots import plot_protein_distributions, plot_protein_westernblots, empty_figure, plot_dimension_reduction_result2d, update_bubble_legend
from RDPMSpecIdentifier.visualize.callbacks.modalCallbacks import FILEEXT
from RDPMSpecIdentifier.visualize.colorSelection import _color_theme_modal, _modal_color_selection, _color_selection
from RDPMSpecIdentifier.visualize import BOOTSH5, BOOTSROW
from RDPMSpecIdentifier.visualize.callbacks.colorCallbacks import *

import plotly.io as pio
import copy

dash.register_page(__name__, path='/figure_factory')

logger = logging.getLogger(__name__)

pio.templates["FFDefault"] = copy.deepcopy(pio.templates["plotly_white"])

pio.templates["FFDefault"].update(
    {
        "layout": {
            # e.g. you want to change the background to transparent
            "paper_bgcolor": "rgba(255,255,255,1)",
            "plot_bgcolor": " rgba(0,0,0,0)",
            "font": dict(color="black"),
            "xaxis": dict(linecolor="black", showline=True),
            "yaxis": dict(linecolor="black", showline=True),
            "coloraxis": dict(colorbar=dict(outlinewidth=1, outlinecolor="black"))
        }
    }
)


def _arg_x_and_y(input_id_x, input_id_y, arg, d_type, default_x, default_y):
    if isinstance(default_x, int):
        step = 1
    elif isinstance(default_x, float):
        step = 0.01
    else:
        step = None
    div = [
        html.Div(
            html.Span(arg, style={"text-align": "center"}),
            className="col-4 col-md-2 justify-content-center align-self-center py-1"
        ),
        html.Div(
            html.Div(
            [
                html.Div(
                    html.Span("X", style={"text-align": "center"}),

                    className="col-1 p-0 align-self-center"
                ),
                html.Div(
                    dcc.Input(
                        style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white",
                               "text-align": "center"},
                        id=input_id_x,
                        className="text-align-center",
                        value=default_x,
                        type=d_type,
                        step=step,
                        persistence=True,
                        persistence_type="session"
                    ),
                    className="col-4 p-0"
                ),
                html.Div(
                    html.Span("Y", style={"text-align": "center"}),

                    className="col-1 p-0 align-self-center"
                ),
                html.Div(
                    dcc.Input(
                        style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white",
                               "text-align": "center"},
                        id=input_id_y,
                        className="text-align-center",
                        value=default_y,
                        type=d_type,
                        step=step,
                        persistence=True,
                        persistence_type="session"
                    ),
                    className="col-4 p-0 "
                )

            ],
                className="row m-0 p-0 justify-content-between"
            ),
            className="col-8 col-md-4 justify-content-center text-align-center align-self-center py-1"
        ),
    ]
    return div

def _args_and_name(input_id, arg, d_type, default):
    if isinstance(default, int):
        step = 1
    elif isinstance(default, float):
        step = 0.01
    else:
        step = None
    div = [
            html.Div(
                html.Span(arg, style={"text-align": "center"}),
                className="col-4 col-md-2 justify-content-center align-self-center py-1"
            ),
            html.Div(
                dcc.Input(
                    style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white",
                           "text-align": "center"},
                    id=input_id,
                    className="text-align-center",
                    value=default,
                    type=d_type,
                    step=step,
                    persistence=True,
                    persistence_type="session",
                ),
                className="col-8 col-md-4 justify-content-center text-align-center py-1"
            )
        ]
    return div

def _arg_and_dropdown(arg, dd_list, default, input_id):
    div = [
        html.Div(
            html.Span(arg, style={"text-align": "center"}),
            className="col-4 col-md-2 justify-content-center align-self-center py-1 "
        ),
        html.Div(
            dcc.Dropdown(
                dd_list, default,
                className="justify-content-center",
                id=input_id,
                clearable=False,
                persistence=True,
                persistence_type="session"

            ),
            className="col-8 col-md-4 justify-content-center text-align-center py-1"
        )
    ]
    return div


def _distribution_settings():
    data = html.Div(
        [
            html.Div(html.H5("General", className="align-text-center"), className="col-12 justify-content-center px-0 align-items-center"),
            *_arg_and_dropdown(
                "Template",
                ["FFDefault"] + [template for template in list(pio.templates) if template != "FFDefault"],
                "FFDefault", "template-dd"
            ),
            *_arg_and_dropdown("Name Col", ["RDPMSpecID"], "RDPMSpecID", "displayed-column-dd"),
            html.Div(html.H5("Plot style"), className=BOOTSH5),
            *_args_and_name("download-width", "Width [px]", "number", 800),
            *_args_and_name("download-height", "Height [px]", "number", 500),
            *_args_and_name("download-marker-size", "Marker Size", "number", 8),
            *_args_and_name("download-line-width", "Line Width", "number", 3),
            *_args_and_name("download-grid-width", "Grid Width", "number", 1),
            *_args_and_name("v-space", "Vertical Space", "number", 0.01),
            *_arg_x_and_y("legend1-x", "legend1-y", "Legend Pos", "number", 0., 1.),
            *_arg_x_and_y("legend2-x", "legend2-y", "Legend2 Pos", "number", 0., 1.05),
            *_arg_x_and_y("x-axis-width", "y-axis-width", "Axis width", "number", 1, 1),
            *_arg_x_and_y("d-x-tick", "d-y-tick", "Axid dtick", "number", 1., 0.1),
            *_arg_x_and_y("zeroline-x-width", "zeroline-y-width", "Zeroline", "number", 1, 0),

        ],
        className=BOOTSROW,
        id="distribution-settings"
    )
    return data


def _font_settings():
    data = html.Div(
        [

            html.Div(html.H5("Fonts"), className=BOOTSH5),
            *_args_and_name("legend-font-size", "Legend", "number", 14),
            *_args_and_name("axis-font-size", "Axis", "number", 18),

        ],
        className=BOOTSROW,
        id="distribution-settings"
    )
    return data


def _bubble_legend_settings():
    data = html.Div(
        [

            html.Div(html.H5("Bubble Legend"), className=BOOTSH5),
            *_args_and_name("legend-start", "Legend Start", "number", 0.25),
            *_args_and_name("legend-spread", "Legend Spread", "number", 0.12),

        ],
        className=BOOTSROW,
        id="bubble-legend-settings"
    )
    return data


def _figure_type():
    div = html.Div(
        [
            html.Div(html.H5("Figure Type"), className=BOOTSH5),
            html.Div(
                [
                    dbc.RadioItems(
                        options=[
                            {'label': 'Distribution', 'value': 0},
                            {'label': 'Westernblot', 'value': 2},
                            {'label': 'Dimension Reduction', 'value': 3},
                        ],
                        value=0,
                        className="d-flex justify-content-around radio-items row",
                        labelCheckedClassName="checked-radio-text",
                        inputCheckedClassName="checked-radio-item",
                        id="plot-type-radio-ff",
                        persistence_type="session",
                        persistence=True
                    ),
                ],
                className="col-12 my-2"
            ),
        ],

        className=BOOTSROW
    )
    return div

def _id_selector():
    div = html.Div(
        [
            html.Div(
                [
                    html.H5(
                        [
                            "RDPMSpecIDs",
                            html.I(className="fas fa-question-circle fa px-2", id="ff-select-tip"),
                            dbc.Tooltip("You can also select IDS via checkboxes in the table on the analysis page",
                                        target="ff-select-tip"),
                        ]
                    ),

                ],
                className="col-12 justify-content-center px-0"),
            html.Div(
                dcc.Dropdown(
                    [],
                    className="justify-content-center",
                    id="protein-selector-ff",
                    clearable=True,
                    multi=True

                ),
                className="col-12"
            )
        ],

        className=BOOTSROW
    )
    return div


def _file_format():
    div = html.Div(
        [
            html.Div(
                html.H5("File Format"),
                className="col-12 px-0"
            ),
            html.Div(
                [

                    dbc.RadioItems(
                        options=[
                            {'label': 'SVG', 'value': "svg"},
                            {'label': 'PNG', 'value': "png"},
                        ],
                        value="svg",
                        inline=True,
                        className="d-flex justify-content-around radio-items",
                        labelCheckedClassName="checked-radio-text",
                        inputCheckedClassName="checked-radio-item",
                        id="filetype-selector-ff",
                        persistence_type="session",
                        persistence=True
                    ),
                ],
                className="col-10 my-2"
            ),
        ],
        className="row justify-content-center px-4 px-md-4"
    )
    return div


def figure_factory_layout():
    layout = html.Div(
        [
            dcc.Store("current-image"),
            _color_theme_modal(2),
            _modal_color_selection("primary-2"),
            _modal_color_selection("secondary-2"),
            html.Div(
                html.Div(
                    [
                        html.Div(
                            html.Div(
                                html.H4("Figure Export"),
                                className="col-12"),
                            className="row justify-content-center"
                        ),


                        html.Div(
                            html.Div(
                                [
                                    _id_selector()

                                ], className="col-12"
                            ),

                            className="row justify-content-center"
                        ),
                        html.Div(
                            html.Div(
                                [
                                    _figure_type()

                                ], className="col-12"
                            ),

                            className="row justify-content-center"
                        ),

                        html.Div(
                            html.Div(
                                [
                                    _distribution_settings(),
                                    _font_settings(),
                                    _bubble_legend_settings()

                                ], className="col-12"
                            ),

                            className="row justify-content-center"
                        ),

                        html.Div(
                            html.Div(
                                [
                                    _file_format()

                                ], className="col-12"
                            ),

                            className="row justify-content-center"
                        ),
                        _color_selection(),
                        html.Div(
                            [
                                html.Div(
                                    html.Button(
                                        'Default Settings', id='ff-default', n_clicks=0,
                                        className="btn btn-primary",
                                    ),
                                    className="col-6 col-lg-3 justify-content-center text-align-center",
                                ),
                                html.Div(
                                    [
                                        html.Span("Filename", className="align-self-center"),
                                        dcc.Input(
                                            style={"width": "100%", "height": "70%", "border-radius": "5px",
                                                   "color": "white",
                                                   "text-align": "center"},
                                            id="download-filename",
                                            className="text-align-center align-self-center mx-2",
                                            value="Image.svg",
                                            type="text",
                                            persistence=True,
                                            persistence_type="session",
                                        ),
                                        html.Button(
                                            'Download', id='ff-download', n_clicks=0,
                                            className="btn btn-primary",
                                        ),
                                    ],


                                    className="col-12 col-lg-9 d-flex"
                                ),
                            ],
                            className=BOOTSROW + " justify-content-between"
                        )
                    ],
                    className="databox p-2"
                ),
                className="col-12 col-lg-6 py-2 px-1"

            ),
            html.Div(
                html.Div(
                    [
                        html.Div(
                            html.H4("Preview", className="col-10 col-md-6 mt-1"),
                            className="row justify-content-center"
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [],
                                    className="col-12 col-md-11 m-2 py-2", id="figure-factory-download-preview",
                                    style={
                                        "overflow-x": "auto",
                                        "background-color": "white",
                                        "border-radius": "10px",
                                        "box-shadow": "inset black 0px 0px 10px",
                                        "border": "1px solid black"

                                    }
                                ),
                            ],
                            className="row justify-content-center px-4 px-md-4",
                        ),
                    ],
                    className="databox"
                ),
                className="col-12 col-lg-6 px-1 py-2"
            ),
            dcc.Download(id="download-ff-image"),

        ],

        className="row"
    )
    return layout

layout = figure_factory_layout()


@callback(
    Output("protein-selector-ff", "options"),
    Output("protein-selector-ff", "value"),
    Input("data-store", "data"),
    State("current-row-ids", "data"),

)
def update_selected_proteins(rdpmsdata: RDPMSpecData, current_row_ids):
    if rdpmsdata is None:
        raise PreventUpdate
    else:
        if current_row_ids is not None:
            value = list(rdpmsdata.df.loc[current_row_ids, "RDPMSpecID"])
        else:
            value = dash.no_update

        return list(rdpmsdata.df["RDPMSpecID"]), value


@callback(
    Output("download-marker-size", "value", allow_duplicate=True),
    Output("download-line-width", "value", allow_duplicate=True),
    Output("download-grid-width", "value"),
    Output("d-x-tick", "value"),
    Output("d-y-tick", "value"),
    Output("download-height", "value"),
    Output("v-space", "value"),
    Output("x-axis-width", "value"),
    Output("y-axis-width", "value"),
    Output("template-dd", "value"),
    Output("zeroline-x-width", "value"),
    Output("zeroline-y-width", "value"),
    Output("legend1-x", "value", allow_duplicate=True),
    Output("legend1-y", "value", allow_duplicate=True),
    Output("legend2-x", "value", allow_duplicate=True),
    Output("legend2-y", "value", allow_duplicate=True),
    Input("ff-default", "n_clicks"),
    State("plot-type-radio-ff", "value"),
    State("current-row-ids", "data"),
    prevent_initial_call=True
)
def apply_default_settings(clicks, plot_type, selected_proteins):
    if clicks is None:
        raise PreventUpdate
    m_size = line_width = grid_width = dtickx = dticky = height = vspace = xaxisw = yaxisw = dash.no_update
    l1x = l2x = l1y = l2y = None
    if plot_type == 0:
        m_size = 5
        line_width = 3
        grid_width = 1
        dtickx = 1
        dticky = None
        height = max(100 * len(selected_proteins), 400)
        vspace = 0.01
        xaxisw = yaxisw = 1
        l1x = l2x = 0
        l1y = 1.01
        l2y = 1.1
    elif plot_type == 2:
        grid_width = 0
        dtickx = 1
        vspace = 0.01
        height = max(100 * len(selected_proteins), 400)
        xaxisw = yaxisw = 1
        l1x = 0
        l1y = 1

    elif plot_type == 3:
        grid_width = 0
        l1x = 1.01
        l1y = 0.85

        dtickx = 2
        dticky = None
        height = 500
        xaxisw = yaxisw = 1

    else:
        pass

    return m_size, line_width, grid_width, dtickx, dticky, height, vspace, xaxisw, yaxisw, "FFDefault", 0, 0, l1x, l1y, l2x, l2y



@callback(
    Output("current-row-ids", "data", allow_duplicate=True),
    Input("protein-selector-ff", "value"),
    State("data-store", "data"),

)
def update_row_ids(values, rdpmsdata):
    if rdpmsdata is None or values is None:
        raise PreventUpdate
    values = list(rdpmsdata.df[rdpmsdata.df["RDPMSpecID"].isin(values)].index)
    return values


@callback(
    Output("displayed-column-dd", "options"),
    Input("data-store", "data"),
)
def update_selectable_columns(rdpmsdata):
    if rdpmsdata is None:
        raise PreventUpdate
    return list(set(rdpmsdata.extra_df) - set(rdpmsdata.score_columns))

@callback(
    Output("current-image", "data"),
    Output("download-marker-size", "value"),
    Output("download-marker-size", "disabled"),
    Output("download-line-width", "value"),
    Output("download-line-width", "disabled"),
    Output("legend1-x", "value"),
    Output("legend1-y", "value"),
    Output("bubble-legend-settings", "style"),
    Input("protein-selector-ff", "value"),
    Input("primary-color", "data"),
    Input("secondary-color", "data"),
    Input("plot-type-radio-ff", "value"),
    Input("displayed-column-dd", "value"),
    Input("v-space", "value"),
    State("data-store", "data"),
    State("unique-id", "data"),
    State("bubble-legend-settings", "style"),

)
def update_download_state(keys, primary_color, secondary_color, plot_type, displayed_col, vspace, rdpmsdata: RDPMSpecData, uid, bubble_style):
    logger.info(f"selected keys: {keys}")
    if plot_type != 3:
        if not keys:
            raise PreventUpdate
    proteins = rdpmsdata.df[rdpmsdata.df.loc[:, "RDPMSpecID"].isin(keys)].index
    logger.info(f"selected proteins: {proteins}")
    bubble_style = bubble_style if bubble_style is not None else {}
    bubble_style["display"] = "none"
    colors = primary_color, secondary_color
    if rdpmsdata.norm_array is None:
        fig = empty_figure(annotation="Data not normalized yet.<br> Visit Analysis Page first.")
        settings = (dash.no_update for _ in DEFAULT_WESTERNBLOT_SETTINGS)
    else:
        if plot_type == 2:
            fig = plot_protein_westernblots(keys, rdpmsdata, colors=colors, title_col=displayed_col, vspace=vspace)
            settings = DEFAULT_WESTERNBLOT_SETTINGS
        elif plot_type == 3:
            if rdpmsdata.current_embedding is not None:
                keys = rdpmsdata.df[rdpmsdata.df.loc[:, "RDPMSpecID"].isin(keys)].index
                fig = plot_dimension_reduction_result2d(
                    rdpmsdata,
                    colors=colors,
                    highlight=keys,
                    clusters=rdpmsdata.df["Cluster"] if "Cluster" in rdpmsdata.df else None,
                    marker_max_size=40,
                    second_bg_color="white",
                    bubble_legend_color="black",
                    sel_column=displayed_col
                )

                fig.update_xaxes(mirror=True, row=2)
                fig.update_yaxes(mirror=True, row=2)
            else:
                fig = empty_figure("Distances not Calculated.<br>Go to Analysis Page and click the Get Score Button.")
            settings = DEFAULT_DIMRED_SETTINGS
            bubble_style["display"] = "flex"

        else:
            fig = plot_protein_distributions(keys, rdpmsdata, colors=colors, title_col=displayed_col, vspace=vspace)
            fig.update_xaxes(mirror=True)
            fig.update_yaxes(mirror=True)
            settings = DEFAULT_DISTRIBUTION_SETTINGS
    fig.update_layout(
        margin=dict(b=70, t=20)
    )
    encoded_image = Serverside(fig, key=uid + "_figure_factory")
    return encoded_image, *settings, bubble_style

DEFAULT_DISTRIBUTION_SETTINGS = (5, False, 3, False, 0., 1.01)
DEFAULT_WESTERNBLOT_SETTINGS = (None, True, None, True, 0., 1.01)
DEFAULT_DIMRED_SETTINGS = (None, True, None, True, 1.01, 0.85)

@callback(
    Output("download-filename", "value"),
    Input("filetype-selector-ff", "value"),
    State("download-filename", "value")
)
def update_filename(filetype, filename):
    if filename is None:
        filename = "Image.svg"
    filename = filename.split(".")[0]
    filename = f"{filename}.{filetype}"
    return filename


@callback(
    Output("download-ff-image", "data"),
    Input('ff-download', "n_clicks"),
    State("figure-factory-download-preview", "children"),
    State("download-filename", "value"),
    prevent_initial_call=True
)
def download_image(n_clicks, figure, filename):
    if n_clicks is None:
        raise PreventUpdate
    fig = figure["props"]["src"]
    ret_val = {}
    ret_val["filename"] = filename
    ret_val["content"] = fig.split(",")[-1]
    ret_val["base64"] = True
    return ret_val

@callback(
    Output("figure-factory-download-preview", "children"),
    Input("current-image", "data"),
    Input("filetype-selector-ff", "value"),
    Input("download-width", "value"),
    Input("download-height", "value"),
    Input("download-marker-size", "value"),
    Input("download-line-width", "value"),
    Input("download-grid-width", "value"),
    Input("zeroline-x-width", "value"),
    Input("zeroline-y-width", "value"),
    Input("d-x-tick", "value"),
    Input("d-y-tick", "value"),
    Input("legend1-x", "value"),
    Input("legend1-y", "value"),
    Input("legend2-x", "value"),
    Input("legend2-y", "value"),
    Input("x-axis-width", "value"),
    Input("y-axis-width", "value"),
    Input("template-dd", "value"),
    Input("legend-font-size", "value"),
    Input("axis-font-size", "value"),
    Input("legend-start", "value"),
    Input("legend-spread", "value"),
    State("plot-type-radio-ff", "value"),

)
def update_ff_download_preview(
        currnet_image,
        filetype,
        img_width,
        img_height,
        marker_size,
        line_width,
        grid_width,
        zeroline_x,
        zeroline_y,
        d_x_tick,
        d_y_tick,
        lx,
        ly,
        l2x,
        l2y,
        xaxis_width,
        yaxis_width,
        template,
        legend_font_size,
        axis_font_size,
        legend_start,
        legend_spread,
        plot_type
):
    try:
        img_width = max(min(img_width, 2000), 100)
        img_height = max(min(img_height, 2000), 100)
    except TypeError:
        img_height = 500
        img_width = 800
    grid_width = max(0, grid_width)
    if currnet_image is None:
        raise PreventUpdate
    logger.info(f"Rendering file with width: {img_width} and height {img_height}")
    fig = currnet_image
    if template:
        fig.update_layout(template=template)
        for data in fig.data:
            if "colorbar" in data:
                data.colorbar.update(pio.templates[template]["layout"]["coloraxis"]["colorbar"])
    fig.update_xaxes(dtick=d_x_tick)
    fig.update_yaxes(dtick=d_y_tick)
    fig.update_xaxes(titlefont=dict(size=axis_font_size))
    fig.update_yaxes(titlefont=dict(size=axis_font_size))
    fig.update_xaxes(showline=True if xaxis_width > 0 else False, linewidth=xaxis_width)
    fig.update_yaxes(showline=True if yaxis_width > 0 else False, linewidth=yaxis_width)
    fig.update_annotations(
        font=dict(size=axis_font_size)
    )
    fig.update_xaxes(zeroline=True if zeroline_x > 0 else False, zerolinewidth=zeroline_x,)
    fig.update_yaxes(zeroline=True if zeroline_y > 0 else False, zerolinewidth=zeroline_y,)
    fig.update_yaxes(gridwidth=grid_width, showgrid=True if grid_width else False)
    fig.update_xaxes(gridwidth=grid_width, showgrid=True if grid_width else False)
    fig.update_layout(
        legend2=dict(
            y=l2y,
            x=l2x
        ),
        legend=dict(
            x=lx,
            y=ly
        )
    )
    if marker_size is not None:
        if marker_size > 0:
            fig.update_traces(
                marker=dict(size=marker_size)
            )
        else:
            fig.update_traces(mode="lines")
    if line_width is not None:
        fig.update_traces(
            line=dict(width=max(line_width, 0)
                      )
        )
    fig.update_layout(
        legend=dict(
            font=dict(size=legend_font_size)
        ),
        legend2=dict(
            font=dict(size=legend_font_size)
        )

    )
    if plot_type == 3:
        legend_start = legend_start if legend_start is not None else 0.25
        legend_spread = legend_spread if legend_spread is not None else 0.125
        try:
            fig = update_bubble_legend(
                fig,
                legend_start=legend_start,
                legend_spread=legend_spread,
                second_bg_color=pio.templates[template]["layout"]["paper_bgcolor"],
                bubble_legend_color=pio.templates[template]["layout"]["font"]["color"]
            )
            fig.update_xaxes(showgrid=False, row=1, showline=False, zeroline=False)
            fig.update_yaxes(showgrid=False, row=1, showline=False, zeroline=False)
            fig.update_annotations(
                font=dict(size=legend_font_size)
            )
        except IndexError:
            pass
        fig.update_xaxes(title=dict(font=dict(size=axis_font_size)))
        fig.update_yaxes(title=dict(font=dict(size=axis_font_size)))
    encoded_image = base64.b64encode(fig.to_image(format=filetype, width=img_width, height=img_height)).decode()
    fig = html.Img(
        src=f'{FILEEXT[filetype]},{encoded_image}',
        style={"margin-left": "auto", "margin-right": "auto", "display": "block"}
    )
    return fig



