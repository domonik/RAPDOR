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
import uuid


def gui_wrapper(input, design_matrix, sep, logbase, debug, port, host):
    try:
        df = pd.read_csv(input, sep=sep, index_col=0)
        df.index = df.index.astype(str)
        design = pd.read_csv(design_matrix, sep=sep)

        app.layout = _get_app_layout(df, design, logbase)
        app.run(debug=debug, port=port, host=host)
    except Exception as e:

        TMPDIR.cleanup()
        raise e
    finally:
        TMPDIR.cleanup()



def _gui_wrapper(args):
    gui_wrapper(args.input, args.design_matrix, args.sep, args.logbase, args.debug, args.port, args.host)


def _get_app_layout(intensities: pd.DataFrame, design: pd.DataFrame, logbase: int, ):
    def return_layout():
        json_intentsities = intensities.to_json()
        json_design = design.to_json()
        rdpmsdata = RDPMSpecData(intensities, design, logbase)
        div = html.Div(
            [
                dcc.Store(id="data-store", storage_type="session"),
                dcc.Store(id="tbl-store"),
                dcc.Store(id="design-store", data=json_design),
                dcc.Store(id="intentity-store", data=json_intentsities),
                dcc.Store(id="logbase-store", data=logbase),
                dcc.Store(id="unique-id", storage_type="session"),
                html.Div(id="recomputation"),
                html.Div(
                    _header_layout(),
                    className="row px-0 justify-content-center align-items-center sticky-top"
                ),
                html.Div(
                    distribution_panel(rdpmsdata),
                    className="row px-2 justify-content-center align-items-center"

                ),
                html.Div(id="test-div", style={"display": "none", "height": "0%"}),
                dcc.Tabs(
                    [
                        dcc.Tab(
                            html.Div(
                                _get_table(rdpmsdata=rdpmsdata),
                                className="row px-2 justify-content-center align-items-center",
                                id="protein-table"
                            ),
                            label="Table", className="custom-tab", selected_className='custom-tab--selected'
                        ),
                        dcc.Tab(
                            html.Div(
                                _get_cluster_panel(),
                                className="row px-2 justify-content-center align-items-center"

                            ), label="Clustering", className="custom-tab", selected_className='custom-tab--selected'
                        )
                    ],
                    parent_className='custom-tabs',
                    className='custom-tabs-container',



                ),


                html.Div(
                    [distance_heatmap_box(), selector_box(rdpmsdata)],
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
        return div
    return return_layout


if __name__ == '__main__':
    try:
        import os
        import pandas as pd
        file = os.path.abspath("testData/testFile.tsv")
        assert os.path.exists(file)
        df = pd.read_csv(file, sep="\t", index_col=0)
        df.index = df.index.astype(str)
        design = pd.read_csv(os.path.abspath("testData/testDesign.tsv"), sep="\t")
        logbase = 2

        app.layout = _get_app_layout(df, design, logbase)
        app.run(debug=True, port=8080, host="127.0.0.1", processes=3, threaded=False)
    except Exception as e:
        TMPDIR.cleanup()
        raise e
    finally:
        TMPDIR.cleanup()
