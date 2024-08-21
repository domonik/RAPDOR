import dash_daq as daq
import dash_loading_spinners as dls
from dash import html, dcc
from RAPDOR.plots import empty_figure




def distribution_panel(name):
    distribution_panel = html.Div(
        [
            html.Div(
                [

                    html.Div(
                        [
                            html.Div(

                                className="", id="placeholder"
                            ),
                            html.Div(
                                html.Div(
                                    [
                                        html.Div(html.H5(f"RAPDORid {name}",
                                                         id="protein-id",
                                                         className="align-self-center"),
                                                 className="col-12 d-flex align-items-center justify-content-center", ),

                                    ],
                                    className="row p-1 h-100"
                                ),
                                className="col-12 col-lg-2"
                            ),
                            html.Div(
                                html.Div(
                                    [
                                        html.Div(
                                            dcc.Dropdown(
                                                [], None,
                                                id="additional-header-dd",
                                                style={"font-size": "1.25rem"},
                                                persistence=True,
                                                persistence_type="session"
                                            ),
                                            className="col-6",

                                        ),

                                        html.Div(html.H5(
                                            "",
                                            id="additional-header",
                                            className="align-self-center",
                                            style={"text-align": "center", "white-space": "nowrap",
                                                   "overflow-x": "hidden",
                                                   "text-overflow": "ellipsis"}),
                                            className="col-6 d-flex justify-content-center"),

                                    ],
                                    className="row p-1 h-100"
                                ),
                                className="col-12 col-lg-3"
                            ),

                            html.Div(
                                html.Div(
                                    [
                                        html.Div(
                                            html.Span("Summary", className="align-self-center"),
                                            className="col-5 d-flex align-items-bottom justify-content-end"
                                        ),
                                        html.Div(

                                            daq.BooleanSwitch(
                                                label='',
                                                labelPosition='left',
                                                color="var(--primary-color)",
                                                on=False,
                                                id="replicate-mode",
                                                className="align-self-center",

                                            ),
                                            className="col-2 d-flex align-items-center justify-content-center"
                                        ),
                                        html.Div(
                                            html.Span("Replicates", className="align-self-center"),
                                            className="col-5 d-flex align-items-bottom justify-content-start"
                                        ),

                                    ],
                                    className="row p-1 h-100"
                                ),
                                className="col-12 col-lg-3"
                            ),
                            html.Div(
                                html.Div(
                                    [
                                        html.Div(
                                            html.Span("Normalized", className="align-self-center"),
                                            className="col-5 d-flex align-items-bottom justify-content-end"
                                        ),
                                        html.Div(

                                            daq.BooleanSwitch(
                                                label='',
                                                labelPosition='left',
                                                color="var(--primary-color)",
                                                on=False,
                                                id="raw-plot",
                                                className="align-self-center",

                                            ),
                                            className="col-2 d-flex align-items-center justify-content-center"
                                        ),
                                        html.Div(
                                            html.Span("Raw", className="align-self-center"),
                                            className="col-5 d-flex align-items-bottom justify-content-start"
                                        ),

                                    ],
                                    className="row p-1 h-100"
                                ),
                                className="col-12 col-lg-3"
                            ),

                        ],
                        className="row justify-content-between p-0 pt-1", id="rapdor-id"
                    ),
                    html.Div(
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            dcc.Graph(id="distribution-graph", style={"height": "340px"},
                                                      figure=empty_figure()),
                                            className="col-12"
                                        ),
                                    ],
                                    className="row justify-content-center"
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            dcc.Graph(id="westernblot-graph", style={"height": "70px"},
                                                      figure=empty_figure(), config={'displayModeBar': False}),
                                            className="col-12"
                                        ),
                                        html.Div("Fraction", className="col-12 pt-0",
                                                 style={"text-align": "center", "font-size": "20px"})
                                    ],
                                    className="row justify-content-center pb-2", id="pseudo-westernblot-row"
                                ),
                            ],
                            style={"min-width": "800px", "width": "99%"}
                        ),
                        style={"overflow-x": "auto"}
                     ),



                ],
                className="databox", )
        ],
        className="col-12 px-1 pb-1 justify-content-center", id="distribution-panel"
    )
    return distribution_panel


def distance_heatmap_box():
    heatmap_box = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        dls.RingChase(
                            [
                                html.Div(
                                    html.H4(
                                        "Distance",
                                        id="distance-header"
                                    ),
                                    className="col-12 pb-2"
                                ),
                                html.Div(
                                    dcc.Graph(id="heatmap-graph", style={"height": "370px"}, figure=empty_figure()),
                                    className="col-12"
                                ),

                            ],
                            color="var(--primary-color)",
                            width=200,
                            thickness=20,
                        ),

                       className="row p-2 justify-content-center",
                    ),

                ],
                className="databox", id="heatmap-box-tut"
            )
        ],
        className="col-12 col-md-6 p-1 justify-content-center equal-height-column"
    )
    return heatmap_box
