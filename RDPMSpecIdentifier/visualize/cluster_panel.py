
from dash import dcc, dash_table
from dash import html, ctx


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