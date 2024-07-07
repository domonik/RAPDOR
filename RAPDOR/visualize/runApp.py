from dash import html, dcc
import pandas as pd

from RAPDOR.datastructures import RAPDORData
from RAPDOR.visualize.backends import celery_app, background_callback_manager

from RAPDOR.visualize.appDefinition import app
import dash_extensions.enrich
from RAPDOR.visualize.colorSelection import _color_theme_modal
from RAPDOR.visualize.staticContent import _header_layout, _footer, _tutorial_raptor
import logging

from RAPDOR.visualize import DISPLAY, TUTORIAL_DIALOG
logging.basicConfig()
logger = logging.getLogger("RAPDOR")




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
            rapdordata = RAPDORData(df, design, logbase=logbase)
        else:
            with open(input) as handle:
                jsons = handle.read()
            rapdordata = RAPDORData.from_json(jsons)
    else:
        rapdordata = None

    app.layout = get_app_layout(rapdordata)
    app.run(debug=debug, port=port, host=host)




def _gui_wrapper(args):
    gui_wrapper(args.input, args.design_matrix, args.sep, args.logbase, args.debug, args.port, args.host)


def get_app_layout(rapdordata: RAPDORData = None):
    def return_layout():
        content = rapdordata.to_jsons() if rapdordata is not None else None
        div = html.Div(
            [
                dcc.Location(id='url', refresh="callback-nav"),
                dcc.Store(id="data-store", storage_type="session"),
                dcc.Store(id="data-initial-store", data=content),
                dcc.Store(id="display-mode", data=DISPLAY, storage_type="session"),
                dcc.Store(id="unique-id", storage_type="session"),
                dcc.Store(id="current-protein-id", data=0),
                dcc.Store(id="ff-ids", storage_type="session"),
                dcc.Store(id="sel-col-state", data=[], storage_type="session"),
                dcc.Store(id="current-row-ids", storage_type="session"),
                dcc.Store(id="table-state", storage_type="session"),
                dcc.Store(id="primary-color", storage_type="session", data="rgb(138, 255, 172)"),
                dcc.Store(id="secondary-color", storage_type="session", data="rgb(255, 138, 221)"),
                dcc.Store(id="tutorial-dialog", storage_type="session", data=TUTORIAL_DIALOG),

                html.Div(id="placeholder2"),
                html.Div(id="placeholder3"),
                html.Div(id="placeholder4"),
                html.Div(id="placeholder5"),
                html.Div(id="placeholder6"),
                html.Div(id="placeholder7"),
                html.Div(id="placeholder8"),
                html.Div(id="placeholder9"),
                dcc.Store(id="tut-output"),
                html.Div(id="display-alert"),

                html.Div(
                    _header_layout(),
                    className="row px-0 justify-content-center align-items-center sticky-top"
                ),
                dash_extensions.enrich.page_container,
                html.Div(
                    _footer(),
                    className="row px-3 py-3 mt-auto justify-content-end align-items-start align-self-bottom",
                    id="footer-row",
                    style={
                        "background-color": "var(--databox-color)",
                        "border-color": "black",
                        "border-width": "2px",
                        "border-style": "solid",
                    },
                ),
                html.Div(
                    _tutorial_raptor(),
                    style={"pointer-events": "None", "z-index": "10002"},
                    className="row p-0 m-0 justify-content-end fixed-bottom d-none",
                    id="tut-row"

                ),
                html.Div(
                    id="tut-overlay",
                    className="overlay d-none shadow"
                )


            ],
            className="container-fluid d-flex flex-column"
        )
        return div
    return return_layout


app.register_celery_tasks()

if __name__ == '__main__':
    import os
    import multiprocessing

    file = os.path.abspath("RAPDOR/tests/testData/testFile.tsv")
    assert os.path.exists(file)
    logger.setLevel(logging.INFO)
    design = "RAPDOR/tests/testData/testDesign.tsv"
    logbase = 2
    gui_wrapper(file, design, host="127.0.0.1", port=8090, debug=True, logbase=logbase)
