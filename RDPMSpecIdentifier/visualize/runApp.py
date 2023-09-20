from dash import html, dcc
import pandas as pd

from RDPMSpecIdentifier.datastructures import RDPMSpecData
from RDPMSpecIdentifier.visualize.appDefinition import app
import dash_extensions.enrich

from RDPMSpecIdentifier.visualize.staticContent import _header_layout, _footer
import logging

from RDPMSpecIdentifier.visualize import DISPLAY

logging.basicConfig()
logger = logging.getLogger("RDPMSpecIdentifier")




def gui_wrapper(
        input: str = None,
        design_matrix: str = None,
        sep: str = "\t",
        logbase: int = None,
        debug: bool = False,
        port: int = 8080,
        host: str = "127.0.0.1"
):
    logger.setLevel(logging.INFO)

    if input is not None:
        if design_matrix is not None:
            df = pd.read_csv(input, sep=sep, index_col=False)
            logger.info(f"loading df:\n{df}")
            design = pd.read_csv(design_matrix, sep=sep)
            rdpmsdata = RDPMSpecData(df, design, logbase=logbase)
        else:
            with open(input) as handle:
                jsons = handle.read()
            rdpmsdata = RDPMSpecData.from_json(jsons)
    else:
        rdpmsdata = None

    app.layout = get_app_layout(rdpmsdata)
    app.run(debug=debug, port=port, host=host)




def _gui_wrapper(args):
    gui_wrapper(args.input, args.design_matrix, args.sep, args.logbase, args.debug, args.port, args.host)


def get_app_layout(rdpmsdata):
    def return_layout():
        content = rdpmsdata.to_jsons() if rdpmsdata is not None else None
        div = html.Div(
            [
                dcc.Location(id='url', refresh="callback-nav"),
                dcc.Store(id="data-store", storage_type="session"),
                dcc.Store(id="data-initial-store", data=content),
                dcc.Store(id="display-mode", data=DISPLAY, storage_type="session"),
                dcc.Store(id="unique-id", storage_type="session"),
                dcc.Store(id="current-protein-id", data=0),
                dcc.Store(id="primary-color", storage_type="session", data="rgb(138, 255, 172)"),
                dcc.Store(id="secondary-color", storage_type="session", data="rgb(255, 138, 221)"),
                html.Div(id="placeholder2"),
                html.Div(id="placeholder3"),
                html.Div(id="display-alert"),
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
    import multiprocessing

    file = os.path.abspath("testData/testFile.tsv")
    assert os.path.exists(file)
    logger.setLevel(logging.INFO)
    design = "testData/testDesign.tsv"
    logbase = 2
    gui_wrapper(file, design, host="0.0.0.0", port=8080, debug=True, logbase=logbase)
