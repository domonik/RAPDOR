import dash_daq as daq
import dash_loading_spinners as dls
from dash import html, dcc
from RDPMSpecIdentifier.plots import empty_figure




def distribution_panel(name):
    distribution_panel = html.Div(
        [
            html.Div(
                [

                    html.Div(
                        [
                            html.Div(

                                className="col-0 col-md-4", id="placeholder"
                            ),
                            html.Div(
                                html.H4(f"Protein {name}", style={"text-align": "center"}, id="protein-id"),
                                className="col-12 col-md-4 justify-content-center align-self-center",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                html.Span("Replicate Mode", className="align-self-center"),
                                                className="col-4 col-md-4 d-flex align-items-bottom justify-content-center"
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
                                                className="col-2 col-md-2 d-flex align-items-center justify-content-center"
                                            ),
                                            html.Div(
                                                html.Button("Download Image", style={"text-align": "center"},
                                                            id="open-modal", className="btn btn-primary"),
                                                className="col-6 col-md-5 justify-content-right align-self-center text-end",
                                            ),

                                        ],
                                        className="row justify-content-right"
                                    ),

                                ],
                                className="col-12 col-md-4"
                            ),

                            dcc.Download(id="download-image"),


                        ],
                        className="row justify-content-center p-2 pt-3"
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Graph(id="distribution-graph", style={"height": "360px"}, figure=empty_figure()),
                                className="col-12"
                            ),
                        ],
                        className="row justify-content-center"
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Graph(id="westernblot-graph", style={"height": "70px"}, figure=empty_figure()),
                                className="col-12"
                            ),
                            html.Div("Fraction", className="col-12 pt-2", style={"text-align": "center", "font-size": "20px"})
                        ],
                        className="row justify-content-center pb-3"
                    ),

                ],
                className="databox",
            )
        ],
        className="col-12 px-1 pb-1 justify-content-center"
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
                className="databox",
            )
        ],
        className="col-12 col-md-6 p-1 justify-content-center equal-height-column"
    )
    return heatmap_box
