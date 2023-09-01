from dash import html, dcc

from RDPMSpecIdentifier.datastructures import RDPMSpecData
from RDPMSpecIdentifier.visualize.appDefinition import app
import dash_extensions.enrich

from RDPMSpecIdentifier.visualize.staticContent import _header_layout, _footer
import logging

logging.basicConfig()
logger = logging.getLogger("RDPMSpecIdentifier")


def gui_wrapper(input, design_matrix, sep, logbase, debug, port, host):
    if input is not None:
        df = pd.read_csv(input, sep=sep, index_col=0)
        design = pd.read_csv(design_matrix, sep=sep)
        rdpmsdata = RDPMSpecData(df, design, logbase=logbase)
    else:
        rdpmsdata = None

    app.layout = _get_app_layout(rdpmsdata)
    app.run(debug=debug, port=port, host=host)




def _gui_wrapper(args):
    gui_wrapper(args.input, args.design_matrix, args.sep, args.logbase, args.debug, args.port, args.host)


def _get_app_layout(rdpmsdata):
    def return_layout():
        content = rdpmsdata.to_jsons() if rdpmsdata is not None else None
        div = html.Div(
            [
                dcc.Location(id='url', refresh="callback-nav"),
                dcc.Store(id="data-store", storage_type="session"),
                dcc.Store(id="data-initial-store", data=content),
                dcc.Store(id="tbl-store"),
                dcc.Store(id="backup"),
                dcc.Store(id="unique-id", storage_type="session"),
                dcc.Store(id="current-protein-id", data=0),
                dcc.Store(id="primary-color", storage_type="session", data="rgb(138, 255, 172)"),
                dcc.Store(id="secondary-color", storage_type="session", data="rgb(255, 138, 221)"),
                html.Div(id="placeholder2"),
                html.Div(id="placeholder3"),
                html.Div(
                    _header_layout(),
                    className="row px-0 justify-content-center align-items-center sticky-top"
                ),
                dash_extensions.enrich.page_container,
                html.Div(
                    _footer(),
                    className="row px-3 py-3 mt-auto justify-content-end align-items-center align-self-bottom",
                    style={
                        "background-color": "var(--databox-color)",
                        "border-color": "black",
                        "border-width": "2px",
                        "border-style": "solid",
                    },
                ),


            ],
            className="container-fluid d-flex flex-column"
        )
        return div
    return return_layout


if __name__ == '__main__':
    import os
    import pandas as pd
    import multiprocessing

    file = os.path.abspath("testData/testFile.tsv")
    assert os.path.exists(file)
    logger.setLevel(logging.INFO)
    df = pd.read_csv(file, sep="\t")
    logger.info("Startup")
    design = pd.read_csv(os.path.abspath("testData/testDesign.tsv"), sep="\t")
    logbase = 2
    rdpmsdata = RDPMSpecData(df, design, logbase)
    app.layout = _get_app_layout(rdpmsdata)
    app.run(debug=True, port=8080, host="127.0.0.1", threaded=True)
