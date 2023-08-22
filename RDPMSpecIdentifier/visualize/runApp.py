from dash import html, dcc

from RDPMSpecIdentifier.datastructures import RDPMSpecData
from RDPMSpecIdentifier.visualize.appDefinition import app
from RDPMSpecIdentifier.visualize.clusterAndSettings import _get_cluster_panel, selector_box
from RDPMSpecIdentifier.visualize.distributionAndHeatmap import distribution_panel, distance_heatmap_box
from RDPMSpecIdentifier.visualize.dataTable import _get_table
from RDPMSpecIdentifier.visualize.modals import (
    _modal_image_download,
    _modal_color_selection,
    _modal_hdbscan_cluster_settings,
    _modal_dbscan_cluster_settings,
    _modal_kmeans_cluster_settings,
    _modal_cluster_image_download
)
from RDPMSpecIdentifier.visualize.staticContent import _header_layout, _footer
from RDPMSpecIdentifier.visualize.callbacks.mainCallbacks import *
from RDPMSpecIdentifier.visualize.callbacks.plotCallbacks import * # Don´t delete that. It is needed.
from RDPMSpecIdentifier.visualize.callbacks.tableCallbacks import *
from RDPMSpecIdentifier.visualize.callbacks.modalCallbacks import *
import RDPMSpecIdentifier.visualize.callbacks
import RDPMSpecIdentifier.visualize


def gui_wrapper(input, design_matrix, sep, logbase, debug, port, host):
    global RDPMSDATA
    global data
    RDPMSDATA = RDPMSpecData.from_files(input, design_matrix, sep=sep, logbase=logbase)
    RDPMSpecIdentifier.visualize.RDPMSDATA = RDPMSDATA

    data = RDPMSDATA.df
    _get_app_layout(app)
    app.run(debug=debug, port=port, host=host)


def _gui_wrapper(args):
    gui_wrapper(args.input, args.design_matrix, args.sep, args.logbase, args.debug, args.port, args.host)


def _get_app_layout(dash_app):
    dash_app.layout = html.Div(
        [
            html.Div(id="recomputation"),
            html.Div(
                _header_layout(),
                className="row px-0 justify-content-center align-items-center sticky-top"
            ),
            html.Div(
                distribution_panel(RDPMSDATA),
                className="row px-2 justify-content-center align-items-center"

            ),
            html.Div(id="test-div", style={"display": "none", "height": "0%"}),
            dcc.Tabs(
                [
                    dcc.Tab(
                        html.Div(
                            _get_table(rdpmsdata=RDPMSDATA),
                            className="row px-2 justify-content-center align-items-center",
                            id="protein-table"
                        ),
                        label="Distribution", className="custom-tab", selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        html.Div(
                            _get_cluster_panel(),
                            className="row px-2 justify-content-center align-items-center"

                        ), label="Clustering", className="custom-tab", selected_className='custom-tab--selected'
                    )
                ],
                parent_className='custom-tabs',
                className='custom-tabs-container pt-2',



            ),


            html.Div(
                [distance_heatmap_box(), selector_box(RDPMSDATA)],
                className="row px-2 row-eq-height justify-content-center"
            ),
            html.Div(
                _footer(),
                className="row px-3 py-3 mt-2 justify-content-end align-items-center",
                style={
                    "background-color": "var(--databox-color)",
                    "border-color": "black",
                    "border-width": "2px",
                    "border-style": "solid",
                },
            ),
            _modal_image_download(),
            _modal_cluster_image_download(),
            _modal_color_selection("primary"),
            _modal_color_selection("secondary"),
            _modal_hdbscan_cluster_settings(),
            _modal_dbscan_cluster_settings(),
            _modal_kmeans_cluster_settings(),

        ],
        className="container-fluid"
    )


if __name__ == '__main__':
    import os
    import pandas as pd
    file = os.path.abspath("../../testData/testFile.tsv")
    assert os.path.exists(file)
    df = pd.read_csv(file, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    design = pd.read_csv(os.path.abspath("../../testData/testDesign.tsv"), sep="\t")
    global RDPMSDATA

    RDPMSDATA = RDPMSpecData(df, design, logbase=2)
    RDPMSpecIdentifier.visualize.RDPMSDATA = RDPMSDATA


    data = RDPMSDATA.df
    _get_app_layout(app)
    app.run(debug=True, port=8080, host="127.0.0.1")