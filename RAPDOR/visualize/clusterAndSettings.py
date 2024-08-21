
from dash import dcc, dash_table, html
from dash import html, ctx
import dash_loading_spinners as dls
import dash_daq as daq
import dash_bootstrap_components as dbc
from RAPDOR.datastructures import RAPDORData
from RAPDOR.plots import empty_figure, DEFAULT_COLORS
from RAPDOR.visualize.colorSelection import _color_theme_modal, _modal_color_selection, _color_selection
from RAPDOR.visualize import BOOTSH5, BOOTSROW, MAX_KERNEL_SLIDER

DISABLED_CLUSERING = "Clustering is disabled in the current version but we might reactivate it at some point"

def _get_cluster_panel(disabled: bool = False):
    panel = html.Div(
        [
            html.Div(
                html.Div(
                    [
                        dcc.Store(id="run-clustering"),
                        html.Div(
                            html.Div(
                                dls.RingChase(
                                    [
                                        dcc.Store(id="plot-dim-red", data=False),
                                        dcc.Graph(id="cluster-graph", figure=empty_figure(),
                                                  style={"min-width": "800px", "height": "400px"}),

                                    ],
                                    color="var(--primary-color)",
                                    width=200,
                                    thickness=20,
                                ),
                                style={"overflow-x": "auto"}, className="m-2"
                            ),
                            className="col-12 col-md-7"
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            html.Span(
                                                [
                                                    "Plot Type",
                                                ],
                                                style={"text-align": "center"}
                                            ),
                                            className="col-3 col-md-3 justify-content-center align-self-center"
                                        ),
                                        html.Div(
                                            dcc.Dropdown(
                                                ["Bubble Plot", "Distance vs Var"],
                                                value="Bubble Plot",
                                                className="justify-content-center",
                                                id="dataset-plot-type",
                                                clearable=False,

                                            ),
                                            className="col-7 justify-content-center text-align-center"
                                        )
                                    ],
                                    className="row justify-content-center p-2"
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            html.Span("Marker Size", style={"text-align": "center"}),
                                            className="col-10 col-md-3 justify-content-center align-self-center"
                                        ),
                                        html.Div(
                                            dcc.Slider(
                                                10, 50, step=1, marks=None,
                                                value=40,
                                                tooltip={"placement": "bottom", "always_visible": True},
                                                className="justify-content-center",
                                                id="cluster-marker-slider",
                                            ),
                                            className="col-10 col-md-7 justify-content-center",
                                        ),
                                    ],
                                    className="row justify-content-center p-2"
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            html.Span(
                                                "Cutoff Type",
                                                style={"text-align": "center"},
                                                id="plot-cutoff-name"
                                            ),
                                            className="col-3 col-md-3 justify-content-center align-self-center"
                                        ),
                                        html.Div(
                                            dcc.Dropdown(
                                                [],
                                                value=None,
                                                className="justify-content-center",
                                                id="cutoff-type",
                                                clearable=True,

                                            ),
                                            className="col-7 justify-content-center text-align-center"
                                        )
                                    ],
                                    className="row justify-content-center p-2"
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            html.Span("Cutoff Range", style={"text-align": "center"}),
                                            className="col-10 col-md-3 justify-content-center align-self-center"
                                        ),
                                        html.Div(
                                            dcc.RangeSlider(-1, 1, value=[0, 1], allowCross=False, id="cutoff-range",
                                                            tooltip={"placement": "top", "always_visible": True}),
                                            className="col-10 col-md-7 justify-content-center",
                                        ),
                                    ],
                                    className="row justify-content-center p-2"
                                ),
                                html.Div(

                                    [
                                        html.Div(
                                            [
                                                html.Span("Cluster", ),
                                                daq.BooleanSwitch(
                                                    label='',
                                                    labelPosition='left',
                                                    color="var(--primary-color)",
                                                    on=False,
                                                    id="showLFC",
                                                    className="align-self-center px-2",
                                                    disabled=disabled
                                                ),
                                                html.Span("LFC",),

                                            ],
                                            className="col-5 d-flex justify-content-center text-align-center"
                                        ),
                                        html.Div(
                                            [
                                                html.Span("2D", ),

                                                daq.BooleanSwitch(
                                                    label='',
                                                    labelPosition='left',
                                                    color="var(--primary-color)",
                                                    on=False,
                                                    id="3d-plot",
                                                    className="align-self-center px-2",
                                                    persistence=True,
                                                    disabled=disabled

                                                ),
                                                html.Span("3D", ),

                                            ],

                                            className="col-5 d-flex justify-content-center text-align-center"
                                        ),
                                    ],
                                    className="row justify-content-center p-2"
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            html.Span(
                                                [
                                                    "Cluster Method",
                                                    html.I(className="fas fa-question-circle fa px-2",
                                                           id="cluster-select-tip"),
                                                    dbc.Tooltip(
                                                        DISABLED_CLUSERING,
                                                        target="cluster-select-tip"),
                                                    dbc.Tooltip(
                                                        DISABLED_CLUSERING,
                                                        target="cluster-method"),
                                                    dbc.Tooltip(
                                                        DISABLED_CLUSERING,
                                                        target="cluster-adjust"),
                                                ],
                                                style={"text-align": "center"}
                                            ),
                                            className="col-3 col-md-3 justify-content-center align-self-center"
                                        ),
                                        html.Div(
                                            dcc.Dropdown(
                                                ["HDBSCAN", "DBSCAN", "K-Means", "None"],
                                                className="justify-content-center",
                                                id="cluster-method",
                                                clearable=False,
                                                disabled=True

                                            ),
                                            className="col-7 justify-content-center text-align-center"
                                        )
                                    ],
                                    className="row justify-content-center p-2"
                                ),
                                html.Div(
                                    html.Div(
                                        html.Button('Adjust Cluster Settings', id='adj-cluster-settings', n_clicks=0, disabled=True,
                                                    className="btn btn-primary", style={"width": "100%"}),
                                        className="col-10 justify-content-center text-align-center", id="cluster-adjust"
                                    ),
                                    className="row justify-content-center p-2"
                                ),
                                # html.Div(
                                #     html.Div(
                                #         html.Button('Download Image', id='cluster-img-modal-btn', n_clicks=0,
                                #                     className="btn btn-primary", style={"width": "100%"}),
                                #         className="col-10 justify-content-center text-align-center"
                                #     ),
                                #     className="row justify-content-center p-2"
                                # ),
                                # dcc.Download(id="download-cluster-image"),

                            ],

                            className="col-md-5 col-12"
                        )
                    ],
                    className="row"
                ),
                className="databox databox-open", id="dim-red-tut"
            )
        ],
        className="col-12 px-1 pb-1 justify-content-center"
    )
    return panel


def selector_box(disabled: bool = False):
    sel_box = html.Div(
        [
            _color_theme_modal(2),
            _modal_color_selection("primary-2"),
            _modal_color_selection("secondary-2"),
            html.Div(
                [
                    html.Div(
                        html.Div(
                            html.H4("Settings", style={"text-align": "center"}),
                            className="col-12 justify-content-center"
                        ),
                        className="row justify-content-center p-2 p-md-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Span("Kernel Size", style={"text-align": "center"}),
                                className="col-12 col-md-4 d-flex p-1 justify-content-center align-self-center"
                            ),
                            html.Div(
                                dcc.Slider(
                                    0, MAX_KERNEL_SLIDER, step=None,
                                    marks={
                                        i: str(i) for i in [0] + list(range(3, MAX_KERNEL_SLIDER + 1, 2))
                                    }, value=3,
                                    className="justify-content-center",
                                    id="kernel-slider",
                                    disabled=disabled
                                ),
                                className="col-12 col-md-8 justify-content-center",
                            ),
                        ],
                        className=BOOTSROW, id="kernel-tut"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Span("Distance Method", style={"text-align": "center"}),
                                className="col-12 col-md-4 d-flex p-1 justify-content-center align-self-center"
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    RAPDORData.methods, RAPDORData.methods[0],
                                    className="justify-content-center",
                                    id="distance-method",
                                    clearable=False,
                                    disabled=disabled,
                                    persistence=True,
                                    persistence_type="session"

                                ),
                                className="col-12 col-md-8 justify-content-center",
                            )
                        ],
                        className=BOOTSROW, id="distance-method-tut"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Button('Get Score', id='score-btn', n_clicks=0, className="btn btn-primary",
                                            style={"width": "100%"}, disabled=disabled),
                                className="col-6 justify-content-center text-align-center"
                            ),
                            html.Div(
                                html.Button('Rank Table', id='rank-btn', n_clicks=0, className="btn btn-primary",
                                            disabled=disabled,
                                            style={"width": "100%"}),
                                className="col-6 justify-content-center text-align-center"
                            ),
                        ],

                        className=BOOTSROW, id="score-rank-tut",
                    ),
                    # html.Div(
                    #     [
                    #         html.Div(
                    #             dcc.Input(
                    #                 style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white",
                    #                        "text-align": "center"},
                    #                 id="distance-cutoff",
                    #                 placeholder="Distance Cutoff",
                    #                 className="text-align-center",
                    #                 type="number",
                    #                 min=0,
                    #                 disabled=disabled
                    #             ),
                    #             className="col-4 text-align-center align-items-center"
                    #         ),
                    #         html.Div(
                    #             html.Button('Peak T-Tests', id='local-t-test-btn', n_clicks=0,
                    #                         className="btn btn-primary",
                    #                         style={"width": "100%"}, disabled=disabled),
                    #             className="col-8 justify-content-center text-align-center"
                    #         ),
                    #     ],
                    #     className=BOOTSROW
                    # ),
                    # html.Div(
                    #     [
                    #         html.Div(
                    #             dcc.Input(
                    #                 style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white", "text-align": "center"},
                    #                 id="permanova-permutation-nr",
                    #                 placeholder="Number of Permutations",
                    #                 className="text-align-center",
                    #                 type="number",
                    #                 min=1,
                    #                 disabled=disabled
                    #             ),
                    #             className="col-4 text-align-center align-items-center"
                    #         ),
                    #         html.Div(
                    #             html.Button('Run PERMANOVA', id='permanova-btn', n_clicks=0,
                    #                         className="btn btn-primary",
                    #                         style={"width": "100%"}, disabled=disabled),
                    #             className="col-8 justify-content-center text-align-center"
                    #         ),
                    #         html.Div(
                    #             id="alert-div",
                    #             className="col-12"
                    #         )
                    #
                    #     ],
                    #     className=BOOTSROW
                    # ),
                    html.Div(
                                html.Div(
                                    id="alert-div",
                                    className="col-12"
                                ),
                        className=BOOTSROW
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Input(
                                    style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white",
                                           "text-align": "center"},
                                    id="anosim-permutation-nr",
                                    placeholder="Number of Permutations",
                                    className="text-align-center",
                                    type="number",
                                    min=1,
                                    disabled=disabled
                                ),
                                className="col-4 text-align-center align-items-center"
                            ),
                            html.Div(
                                html.Button('Run ANOSIM', id='anosim-btn', n_clicks=0,
                                            className="btn btn-primary", disabled=disabled,
                                            style={"width": "100%"}),
                                className="col-8 justify-content-center text-align-center"
                            ),

                        ],
                        className=BOOTSROW, id="anosim-tut"
                    ),

                    html.Div(
                        [
                            html.Div(
                                html.Button('Export JSON', id='export-pickle-btn', n_clicks=0, className="btn btn-primary",
                                            style={"width": "100%"}),
                                className="col-6 justify-content-center text-align-center"
                            ),
                            dcc.Download(id="download-pickle"),
                            html.Div(
                                html.Button('Export TSV', id='export-btn', n_clicks=0, className="btn btn-primary",
                                            style={"width": "100%"}),
                                className="col-6 justify-content-center text-align-center"
                            ),
                            dcc.Download(id="download-dataframe-csv"),
                        ],

                        className=BOOTSROW, id="export-tut"
                    ),
                    html.Div(
                        html.Div(
                            _color_selection(),
                            className="col-12"),
                        className="row justify-content-center pb-2", id="color-tut"
                    ),
                ],
                className="databox justify-content-center", id="selector-box-tut"
            )
        ],
        className="col-12 col-md-6 p-1 justify-content-center equal-height-column"
    )
    return sel_box
