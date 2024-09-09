

import dash
import pandas as pd
from dash import Output, Input, html, ctx, dcc

from dash.exceptions import PreventUpdate

from dash_extensions.enrich import Serverside, State, callback
from RAPDOR.datastructures import RAPDORData
import uuid
import dash_bootstrap_components as dbc
from RAPDOR.visualize import DISPLAY, DEFAULT_COLUMNS

import logging

logger = logging.getLogger(__name__)


def remove_list_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list


@callback(
    Output("unique-id", "data"),
    Output("data-store", "data", allow_duplicate=True),
    Input("unique-id", "data"),
    State("data-store", "data"),
    State("data-initial-store", "data")
)
def assign_session_identifier(uid, data, initial_data):
    logger.info(f"data is {data} uid is {uid}")
    rdata = dash.no_update
    if uid is None:
        uid = str(uuid.uuid4())
    if data is None :
        if initial_data is not None:
            rdata = Serverside(RAPDORData.from_json(initial_data), key=uid)
            logger.info("Setting initial from initital data")

        elif DISPLAY:
            rdata = Serverside(None, key=uid)
            logger.info("Setting initial from initital data")

    return uid, rdata

#
# @callback(
#     Output("data-store", "data", allow_duplicate=True),
#     Output("kernel-slider", "value"),
#     Output("distance-method", "value"),
#     Output('cluster-feature-slider', 'value'),
#     State("data-initial-store", "data"),
#     Input("data-store", "data"),
#
# )
# def load_state(uid, data, saved):
#     logger.info("Loading Data")
#     if saved is None:
#         logger.info("Loading from initial state")
#         rapdor = RAPDORData.from_json(data)
#     else:
#         logger.info("Loading from saved state")
#         rapdor = saved
#     kernel = rapdor.state.kernel_size if rapdor.state.kernel_size is not None else 3
#     dm = rapdor.state.distance_method if rapdor.state.distance_method is not None else dash.no_update
#     cf_slider = rapdor.state.cluster_kernel_distance if rapdor.state.cluster_kernel_distance is not None else dash.no_update
#     return Serverside(rapdor, key=uid), kernel, dm, uid, cf_slider

@callback(
    Output("kernel-slider", "value"),
    Output("cluster-method", "value"),
    Output("table-selector", "value", allow_duplicate=True),
    Output("additional-header-dd", "options"),
    Output("additional-header-dd", "value"),
    Output("kernel-slider", "disabled"),
    Output("distance-method", "value"),
    Input("unique-id", "data"),
    Input("refresh-btn", "n_clicks"),
    State("data-store", "data"),
    State("additional-header-dd", "value"),
    State("sel-col-state", "data"),
)
def load_initital_state(uid, pathname, rapdordata: RAPDORData, selected_ad_header, sel_col_state):
    logger.info(f" {ctx.triggered_id} triggered Setting from state")
    if uid is None:
        logger.info("user id is None. Not setting from state")
        raise PreventUpdate
    if rapdordata is None:
        logger.info("rapdordata is None. Not setting from state")
        raise PreventUpdate
    state = rapdordata.state
    logger.info(f"state: {state}")
    if rapdordata.categorical_fraction:
        kernel_size = 0
        kernel_disabled = True
    else:
        kernel_size = state.kernel_size if state.kernel_size is not None else 3
        kernel_disabled = dash.no_update
    cluster_method = state.cluster_method if state.cluster_method is not None else dash.no_update
    if sel_col_state is None or len(sel_col_state) == 0:
        sel_columns = []
        logger.info("Table dropdown state does not match the selection, will update")
        for name in DEFAULT_COLUMNS:
            if name in rapdordata.extra_df:
                sel_columns.append(name)
        sel_columns = remove_list_duplicates(sel_columns)
    else:
        sel_columns = dash.no_update
    dm = rapdordata.state.distance_method if rapdordata.state.distance_method is not None else dash.no_update
    logger.info(f"Initially Selected Columns: {sel_columns}")
    options = list(set(rapdordata.extra_df) - set(rapdordata.score_columns + rapdordata._id_columns + rapdordata.replicate_info))
    logger.info(selected_ad_header)
    if selected_ad_header is None:
        selected_ad_header = list(rapdordata.extra_df)[0] if "Gene" not in rapdordata.extra_df else "Gene"
    return kernel_size, cluster_method, sel_columns, options, selected_ad_header, kernel_disabled, dm


@callback(
    Output("recomputation", "children"),
    Output("data-store", "data", allow_duplicate=True),
    Output('table-selector', 'value', allow_duplicate=True),
    Output('sel-col-state', 'data', allow_duplicate=True),
    Output('tbl', 'sort_by', allow_duplicate=True),
    Output('tbl', 'filter_query', allow_duplicate=True),
    Input("kernel-slider", "value"),
    Input("distance-method", "value"),
    State("data-store", "data"),
    State("unique-id", "data"),
    State('table-selector', 'value'),
    prevent_initial_call=True
)
def recompute_data(kernel_size, distance_method, rapdordata, uid, selected_columns):
    if rapdordata is None:
        raise PreventUpdate
    if uid is None:
        raise PreventUpdate
    logger.info(f"Normalization triggered via {ctx.triggered_id}")
    eps = 10 if distance_method == "KL-Divergence" else 0  # Todo: Make this optional
    rapdordata: RAPDORData
    if rapdordata.state.kernel_size != kernel_size or rapdordata.state.distance_method != distance_method:
        logger.info(f"Normalizing using method: {distance_method} and eps: {eps}")
        rapdordata.normalize_and_get_distances(method=distance_method, kernel=kernel_size, eps=eps)
        selected_columns = [] if selected_columns is None else selected_columns
        selected_columns = [col for col in selected_columns if col not in rapdordata.score_columns]
        return html.Div(), Serverside(rapdordata, key=uid), selected_columns, selected_columns, [], ""
    logger.info("Data already Normalized")
    raise PreventUpdate

#
# @app.callback(
#     Output("logo-container", "children"),
#     Input("night-mode", "on"),
#     Input("secondary-color", "data"),
# )
# def update_logo(night_mode, color):
#     rep = f"fill:{color}"
#     l_image_text = IMG_TEXT[:COLOR_IDX] + rep + IMG_TEXT[COLOR_END:]
#     if not night_mode:
#         l_image_text = re.sub("fill:#f2f2f2", "fill:black", l_image_text)
#     encoded_img = base64.b64encode(l_image_text.encode())
#     img = 'data:image/svg+xml;base64,{}'.format(encoded_img.decode())
#     return html.Img(src=img, style={"width": "20%", "min-width": "300px"}, className="p-1"),


@callback(
        Output("protein-id", "children"),
        Output("current-protein-id", "data"),
        Output("additional-header", "children"),

    [
        Input('tbl', 'active_cell'),
        Input("test-div", "children"),
        Input("additional-header-dd", "value"),

    ],
    State("data-store", "data"),

)
def update_selected_id(active_cell, test_div, additional_header, rapdordata):
    logger.info(f"{ctx.triggered_id} -- triggered update of selected Protein")
    if rapdordata is None:
        raise PreventUpdate
    if ctx.triggered_id == "tbl" or ctx.triggered_id == "additional-header-dd":
        logger.info(f"active cell is: {active_cell}")
        if active_cell is None:
            active_row_id = None
            protein = None
        else:
            logger.info(f"active cell is: {active_cell}")
            active_row_id = active_cell["row_id"]
            protein = rapdordata.df.loc[active_row_id, "RAPDORid"]

    elif ctx.triggered_id == "test-div":
        logger.info(f"{test_div} - value")
        if test_div is None:
            raise PreventUpdate
        active_row_id = int(test_div)
        active_cell = active_row_id
        protein = rapdordata.df.loc[active_row_id, "RAPDORid"]
    else:
        raise PreventUpdate
    protein = f"Protein {protein}"
    if additional_header:
        if active_cell is not None:
            additional_display = rapdordata.df.loc[active_row_id, additional_header]
            if pd.isna(additional_display):
                additional_display = "Na"
        else:
            additional_display = "Na"
    else:
        additional_display = ""
    return protein, active_row_id, additional_display


@callback(
    Output("download-dataframe-csv", "data"),
    Input("export-btn", "n_clicks"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def download_dataframe(n_clicks, rapdordata: RAPDORData):
    df = rapdordata.extra_df.drop(["id"], axis=1)
    return dcc.send_data_frame(df.to_csv, "RAPDOR.tsv", sep="\t")




@callback(
    Output("download-pickle", "data"),
    Input("export-pickle-btn", "n_clicks"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def download_json(n_clicks, rapdordata):
    ret_val = dict(
        content=rapdordata.to_jsons(),
        filename="RAPDOR.json"
    )

    return ret_val

@callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    Input("url", "pathname"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, url, is_open):
    logger.info(f"{ctx.triggered_id} - triggered side canvas")
    if ctx.triggered_id == "url":
        if is_open:
            logger.info("Closing off-canvas")
            return not is_open
    else:
        if n1:
            return not is_open
    return is_open


@callback(
    Output("display-mode", "data"),
    Output("display-alert", "children"),
    Input("unique-id", "data"),
    State("display-mode", "data"),

)
def display_mode_alert(uid, display_mode):
    if display_mode:
        alert_msg = html.Div(
            [
                html.H3("Display mode", style={"color": "white"}),
                html.Div([
                    "This is ment to inspect pre-analyzed data. You can download the tool for data analysis here: ",
                    html.Br(),
                    html.A("https://github.com/domonik/RAPDOR/releases", target="_blank", href="https://github.com/domonik/RAPDOR/releases", id="displayModeLink", style={"line-break": "anywhere"})
                ])
            ]
        )

        alert_msg = html.Div(
            dbc.Alert(
                alert_msg,
                color="var(--primary-color)",
                dismissable=True,
            ),
            className="p-2 align-items-center, alert-msg col-md-5 col-12",

        )
        return False, alert_msg

    else:
        raise PreventUpdate