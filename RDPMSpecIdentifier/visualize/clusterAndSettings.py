
from dash import dcc, dash_table, html
from dash import html, ctx
import dash_loading_spinners as dls
import dash_daq as daq
from RDPMSpecIdentifier.visualize.staticContent import DEFAULT_COLORS
from RDPMSpecIdentifier.datastructures import RDPMSpecData
from RDPMSpecIdentifier.plots import empty_figure


def _get_cluster_panel(disabled: bool = False):
    panel = html.Div(
        [
            html.Div(
                html.Div(
                    [
                        dcc.Store(id="run-clustering"),
                        html.Div(
                            dls.RingChase(
                                [
                                    dcc.Store(id="plot-dim-red", data=False),
                                    dcc.Graph(id="cluster-graph", figure=empty_figure()),

                                ],
                                color="var(--primary-color)",
                                width=200,
                                thickness=20,
                            ),
                            className="col-12 col-md-7"
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            html.Span("Dimension Reduction", style={"text-align": "center"}),
                                            className="col-3 col-md-3 justify-content-center align-self-center"
                                        ),
                                        html.Div(
                                            dcc.Dropdown(
                                                ["T-SNE", "UMAP", "PCA"], "T-SNE",
                                                className="justify-content-center",
                                                id="dim-red-method",
                                                clearable=False,
                                                disabled=disabled

                                            ),
                                            className="col-7 justify-content-center text-align-center"
                                        )
                                    ],
                                    className="row justify-content-center p-2 pt-5"
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            html.Span("3D", style={"text-align": "center"}),
                                            className="col-3 col-md-3 justify-content-center align-self-center"
                                        ),
                                        html.Div(
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
                                            className="col-7 justify-content-center text-align-center"
                                        )
                                    ],
                                    className="row justify-content-center p-2"
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            html.Span("Feature Kernel Size", style={"text-align": "center"}),
                                            className="col-10 col-md-3 justify-content-center align-self-center"
                                        ),
                                        html.Div(
                                            dcc.Slider(
                                                0, 10, step=1,
                                                value=3,
                                                className="justify-content-center",
                                                id="cluster-feature-slider",
                                                disabled=disabled
                                            ),
                                            className="col-10 col-md-7 justify-content-center",
                                        ),
                                    ],
                                    className="row justify-content-center p-2"
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            html.Span("Cluster Method", style={"text-align": "center"}),
                                            className="col-3 col-md-3 justify-content-center align-self-center"
                                        ),
                                        html.Div(
                                            dcc.Dropdown(
                                                ["HDBSCAN", "DBSCAN", "K-Means", "None"], "HDBSCAN",
                                                className="justify-content-center",
                                                id="cluster-method",
                                                clearable=False,
                                                disabled=disabled

                                            ),
                                            className="col-7 justify-content-center text-align-center"
                                        )
                                    ],
                                    className="row justify-content-center p-2"
                                ),
                                html.Div(
                                    html.Div(
                                        html.Button('Adjust Cluster Settings', id='adj-cluster-settings', n_clicks=0, disabled=disabled,
                                                    className="btn btn-primary", style={"width": "100%"}),
                                        className="col-10 justify-content-center text-align-center"
                                    ),
                                    className="row justify-content-center p-2"
                                ),
                                html.Div(
                                    html.Div(
                                        html.Button('Download Image', id='cluster-img-modal-btn', n_clicks=0,
                                                    className="btn btn-primary", style={"width": "100%"}),
                                        className="col-10 justify-content-center text-align-center"
                                    ),
                                    className="row justify-content-center p-2"
                                ),
                                dcc.Download(id="download-cluster-image"),

                            ],

                            className="col-md-5 col-12"
                        )
                    ],
                    className="row"
                ),
                className="databox databox-open"
            )
        ],
        className="col-12 px-1 pb-1 justify-content-center"
    )
    return panel


def selector_box(disabled: bool = False):
    sel_box = html.Div(
        [
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
                                html.Span("Distance Method", style={"text-align": "center"}),
                                className="col-3 col-md-3 justify-content-center align-self-center"
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    RDPMSpecData.methods, RDPMSpecData.methods[0],
                                    className="justify-content-center",
                                    id="distance-method",
                                    clearable=False,
                                    disabled=disabled

                                ),
                                className="col-7 justify-content-center text-align-center"
                            )
                        ],
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Span("Kernel Size", style={"text-align": "center"}),
                                className="col-10 col-md-3 justify-content-center align-self-center"
                            ),
                            html.Div(
                                dcc.Slider(
                                    0, 5, step=None,
                                    marks={
                                        0: "0",
                                        3: '3',
                                        5: '5',
                                    }, value=3,
                                    className="justify-content-center",
                                    id="kernel-slider",
                                    disabled=disabled
                                ),
                                className="col-10 col-md-7 justify-content-center",
                            ),
                        ],
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        html.Div(
                            html.Button('Get Score', id='score-btn', n_clicks=0, className="btn btn-primary", style={"width": "100%"}, disabled=disabled),
                            className="col-10 justify-content-center text-align-center"
                        ),
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        html.Div(
                            html.Button('Rank Table', id='rank-btn', n_clicks=0, className="btn btn-primary", disabled=disabled,
                                        style={"width": "100%"}),
                            className="col-10 justify-content-center text-align-center"
                        ),
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Input(
                                    style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white",
                                           "text-align": "center"},
                                    id="distance-cutoff",
                                    placeholder="Distance Cutoff",
                                    className="text-align-center",
                                    type="number",
                                    min=0,
                                    disabled=disabled
                                ),
                                className="col-3 text-align-center align-items-center"
                            ),
                            html.Div(
                                html.Button('Peak T-Tests', id='local-t-test-btn', n_clicks=0,
                                            className="btn btn-primary",
                                            style={"width": "100%"}, disabled=disabled),
                                className="col-7 justify-content-center text-align-center"
                            ),
                        ],
                        className="row justify-content-center p-2"
                    ),

                    html.Div(
                        [
                            html.Div(
                                dcc.Input(
                                    style={"width": "100%", "height": "100%", "border-radius": "5px", "color": "white", "text-align": "center"},
                                    id="permanova-permutation-nr",
                                    placeholder="Number of Permutations",
                                    className="text-align-center",
                                    type="number",
                                    min=1,
                                    disabled=disabled
                                ),
                                className="col-3 text-align-center align-items-center"
                            ),
                            html.Div(
                                html.Button('Run PERMANOVA', id='permanova-btn', n_clicks=0,
                                            className="btn btn-primary",
                                            style={"width": "100%"}, disabled=disabled),
                                className="col-7 justify-content-center text-align-center"
                            ),
                            html.Div(
                                id="alert-div",
                                className="col-10"
                            )

                        ],
                        className="row justify-content-center p-2"
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
                                className="col-3 text-align-center align-items-center"
                            ),
                            html.Div(
                                html.Button('Run ANOSIM', id='anosim-btn', n_clicks=0,
                                            className="btn btn-primary", disabled=disabled,
                                            style={"width": "100%"}),
                                className="col-7 justify-content-center text-align-center"
                            ),

                        ],
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Button('Export TSV', id='export-btn', n_clicks=0, className="btn btn-primary",
                                            style={"width": "100%"}),
                                className="col-10 justify-content-center text-align-center"
                            ),
                            dcc.Download(id="download-dataframe-csv"),
                        ],

                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Button('Export JSON', id='export-pickle-btn', n_clicks=0, className="btn btn-primary",
                                            style={"width": "100%"}),
                                className="col-10 justify-content-center text-align-center"
                            ),
                            dcc.Download(id="download-pickle"),
                        ],

                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        [
                            html.Div(
                                html.Span("Select Color Scheme", style={"text-align": "center"}, id="color-scheme"),
                                className="col-4 col-md-4 justify-content-center align-self-center"
                            ),
                            html.Div(
                                html.Button(
                                    '', id='primary-open-color-modal', n_clicks=0, className="btn",
                                    style={"width": "100%", "height": "40px"}
                                ),
                                className="col-3 justify-content-center text-align-center primary-color-div"
                            ),
                            html.Div(
                                html.Button(
                                    '', id='secondary-open-color-modal', n_clicks=0,
                                    className="btn",
                                    style={"width": "100%", "height": "40px"}
                                ),
                                className="col-3 justify-content-center text-align-center primary-color-div"
                            ),

                        ],

                        className="row justify-content-center p-2"
                    ),
                ],
                className="databox justify-content-center"
            )
        ],
        className="col-12 col-md-6 p-1 justify-content-center equal-height-column"
    )
    return sel_box
