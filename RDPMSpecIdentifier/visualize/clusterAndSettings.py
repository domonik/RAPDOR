
from dash import dcc, dash_table, html
from dash import html, ctx

from RDPMSpecIdentifier.visualize.staticContent import DEFAULT_COLORS


def _get_cluster_panel():
    panel = html.Div(
        [
            html.Div(
                html.Div(
                    [
                        html.Div(html.H4("Clustering"), className="col-12 py-2"),
                        html.Div(
                            dcc.Graph(id="cluster-graph"),
                            className="col-12 col-md-7"
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(html.H4("Cluster Settings"), className="col-12 pt-2")
                                    ],
                                    className="row"

                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            html.Span("Dimension Reduction", style={"text-align": "center"}),
                                            className="col-3 col-md-3 justify-content-center align-self-center"
                                        ),
                                        html.Div(
                                            dcc.Dropdown(
                                                ["T-SNE", "UMAP"], "T-SNE",
                                                className="justify-content-center",
                                                id="dim-red-method",
                                                clearable=False

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
                                                0, 5, step=1,
                                                value=3,
                                                className="justify-content-center",
                                                id="cluster-feature-slider"
                                            ),
                                            className="col-10 col-md-7 justify-content-center",
                                        ),
                                    ],
                                    className="row justify-content-center p-2"
                                ),
                                html.Div(
                                    html.Div(
                                        html.Button('Get Low Dim Plot', id='dim-red-btn', n_clicks=0,
                                                    className="btn btn-primary", style={"width": "100%"}),
                                        className="col-10 justify-content-center text-align-center"
                                    ),
                                    className="row justify-content-center p-2"
                                ),
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


def selector_box(rdpmsdata):
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
                                    list(rdpmsdata.methods.keys()), list(rdpmsdata.methods.keys())[0],
                                    className="justify-content-center",
                                    id="distance-method",
                                    clearable=False

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
                                    id="kernel-slider"
                                ),
                                className="col-10 col-md-7 justify-content-center",
                            ),
                        ],
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        html.Div(
                            html.Button('Get Score', id='score-btn', n_clicks=0, className="btn btn-primary", style={"width": "100%"}),
                            className="col-10 justify-content-center text-align-center"
                        ),
                        className="row justify-content-center p-2"
                    ),
                    html.Div(
                        html.Div(
                            html.Button('Rank Table', id='rank-btn', n_clicks=0, className="btn btn-primary",
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
                                ),
                                className="col-3 text-align-center align-items-center"
                            ),
                            html.Div(
                                html.Button('Peak T-Tests', id='local-t-test-btn', n_clicks=0,
                                            className="btn btn-primary",
                                            style={"width": "100%"}),
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
                                    min=1
                                ),
                                className="col-3 text-align-center align-items-center"
                            ),
                            html.Div(
                                html.Button('Run PERMANOVA', id='permanova-btn', n_clicks=0,
                                            className="btn btn-primary",
                                            style={"width": "100%"}),
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
                                    min=1
                                ),
                                className="col-3 text-align-center align-items-center"
                            ),
                            html.Div(
                                html.Button('Run ANOSIM', id='anosim-btn', n_clicks=0,
                                            className="btn btn-primary",
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
                                html.Span("Select Color Scheme", style={"text-align": "center"}, id="color-scheme"),
                                className="col-4 col-md-4 justify-content-center align-self-center"
                            ),
                            html.Div(
                                html.Button(
                                    '', id='primary-open-color-modal', n_clicks=0, className="btn btn-primary",
                                    style={"width": "100%", "height": "40px", "background-color": DEFAULT_COLORS["primary"]}
                                ),
                                className="col-3 justify-content-center text-align-center primary-color-div"
                            ),
                            html.Div(
                                html.Button(
                                    '', id='secondary-open-color-modal', n_clicks=0,
                                    className="btn btn-primary",
                                    style={"width": "100%", "height": "40px", "background-color": DEFAULT_COLORS["secondary"]}
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
