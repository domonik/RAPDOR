import os
import time

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Output, Input, ctx, html
from dash.exceptions import PreventUpdate
from pandas.core.dtypes.common import is_numeric_dtype
from dash.dash_table.Format import Format
from RAPDOR.visualize.dataTable import SELECTED_STYLE, _create_table
from RAPDOR.visualize import DISPLAY
from RAPDOR.visualize.callbacks.mainCallbacks import remove_list_duplicates
from dash_extensions.enrich import Serverside, State, callback
import logging
logger = logging.getLogger(__name__)

MAXPERMUTATIONS = 9999

@callback(
    Output("ff-ids", "data", allow_duplicate=True),
    Input('tbl', 'selected_row_ids'),
    State('data-store', 'data'),

)
def update_ff_ids(selected_columns, rapdordata):
    if selected_columns is None or selected_columns == []:
        raise PreventUpdate
    proteins = rapdordata.df.iloc[list(selected_columns)]["RAPDORid"]

    return proteins


@callback(
    Output("tbl", "selected_rows"),
    Output("current-row-ids", "data", allow_duplicate=True),
    Output("tbl", "selected_row_ids", allow_duplicate=True),
    Input("tbl", "derived_viewport_row_ids"),
    State("tbl", "selected_row_ids"),
    State("current-row-ids", "data"),

)
def update_selection_on_page_switch(vpids, selected_ids, current_ids):
    if vpids is None:
        raise PreventUpdate
    if selected_ids is None:
        selected_ids = []
    logger.info(f"Syncing row Ids {selected_ids}, {current_ids}, {vpids}")
    vpids = np.asarray(vpids)
    selected_ids = list(dict.fromkeys(selected_ids + current_ids)) if current_ids is not None else selected_ids
    selected_ids = np.asarray(selected_ids)
    rows = np.where(np.isin(vpids, selected_ids))[0]
    return rows, selected_ids, selected_ids




@callback(
    Output("tbl", "selected_rows", allow_duplicate=True),
    Output("current-row-ids", "data", allow_duplicate=True),
    Output("tbl", "selected_row_ids", allow_duplicate=True),
    Input("reset-rows-btn", "n_clicks"),
    prevent_initial_call=True
)
def reset_selected_rows(n_clicks):
    if n_clicks is not None:
        return [], [], []
    else: raise PreventUpdate


@callback(
    Output("current-row-ids", "data", allow_duplicate=True),
    Input("tbl", "selected_row_ids"),
    State("current-row-ids", "data"),
    State("tbl", "derived_viewport_row_ids"),

)
def update_current_rows(sel_rows, current_selection, vpids):
    if sel_rows is None or vpids is None:
        raise PreventUpdate
    if current_selection is None:
        current_selection = []
    logger.info(f"selected-row-ids on page {sel_rows}")
    current_selection = [cid for cid in current_selection if cid not in set(vpids)]
    sel_rows = list(dict.fromkeys(sel_rows + current_selection))
    logger.info(f"current-row-ids {sel_rows}")
    return sel_rows

@callback(
    Output("table-state", "data"),
    Input('tbl', "page_current"),
    Input('tbl', 'sort_by'),
    Input('tbl', 'filter_query'),

)
def save_table_state(page_current, sort_by, filter_query):
    tbl_state = {"page_current": page_current, "sort_by": sort_by, "filter_query": filter_query}
    return tbl_state


@callback(
    Output('tbl', "page_current", allow_duplicate=True),
    Output('tbl', 'sort_by'),
    Output('tbl', 'filter_query'),
    Input("tbl", "columns"),
    State("table-state", "data"),

)
def load_table_state(pathname, table_state):
    if table_state is None:
        raise PreventUpdate
    return table_state["page_current"], table_state["sort_by"], table_state["filter_query"]


@callback(
    Output('tbl', 'data'),
    Output('tbl', 'selected_row_ids'),
    Output('tbl', 'active_cell'),
    Output('tbl', 'page_current'),
    Input('tbl', "data"),
    Input('tbl', "page_current"),
    Input('tbl', "page_size"),
    Input('tbl', 'sort_by'),
    Input('tbl', 'filter_query'),
    State('table-selector', 'value'),
    State("current-row-ids", "data"),
    State("data-store", "data")
)
def update_table(table_data, page_current, page_size, sort_by, filter_query, selected_columns, selected_row_ids, rapdordata):
    if rapdordata is None or page_current is None:
        raise PreventUpdate

    if selected_columns is None:
        selected_columns = []
    data = rapdordata.extra_df.loc[:, rapdordata._id_columns + selected_columns]

    if filter_query is not None:
        filtering_expressions = filter_query.split(' && ')
        for filter_part in filtering_expressions:
            col_name, operator, filter_value = split_filter_part(filter_part)

            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                # these operators match pandas series operator method names
                data = data.loc[getattr(data[col_name], operator)(filter_value)]
            elif operator == 'contains':
                filter_value = str(filter_value).split(".0")[0]
                data = data.loc[data[col_name].str.contains(filter_value, case=False).fillna(False)]
            elif operator == 'datestartswith':
                filter_value = str(filter_value).split(".0")[0]

                # this is a simplification of the front-end filtering logic,
                # only works with complete fields in standard format
                data = data.loc[data[col_name].str.startswith(filter_value)]

    if sort_by is not None:
        sort_by = [col for col in sort_by if col["column_id"] in data]
        if len(sort_by):
            data = data.sort_values(
                [col['column_id'] for col in sort_by],
                ascending=[
                    col['direction'] == 'asc'
                    for col in sort_by
                ],
                inplace=False
            )

    if "tbl.page_current" in ctx.triggered_prop_ids or "tbl.sort_by" in ctx.triggered_prop_ids:
        page = page_current
        if len(data) > 0:
            active_cell_out = {'row': 1, 'column': 1, 'column_id': 'RAPDORid', 'row_id': 0}
        else:
            active_cell_out = None
    elif "tbl.filter_query" in ctx.triggered_prop_ids:
        logger.info(page_current)
        page = 0
        if len(data) > 0:
            active_cell_out = {'row': 1, 'column': 1, 'column_id': 'RAPDORid', 'row_id': 0}
        else:
            active_cell_out = None
    else:
        active_cell_out = dash.no_update
        page = page_current
    if page * page_size >= data.shape[0]:
        return_value = pd.DataFrame()
    else:
        return_value = data.iloc[page * page_size: (page_current + 1) * page_size]
        if isinstance(active_cell_out, dict):
            loc = return_value.iloc[0].id
            active_cell_out['row_id'] = loc
    return return_value.to_dict('records'), selected_row_ids, active_cell_out, page

#
# @callback(
#     [
#         Output("tbl", "columns"),
#         Output('tbl', 'data', allow_duplicate=True),
#         Output("alert-div", "children", allow_duplicate=True),
#         Output('tbl', 'sort_by'),
#         Output('data-store', 'data', allow_duplicate=True),
#         Output('run-clustering', 'data', allow_duplicate=True),
#         Output('table-selector', 'value'),
#
#     ],
#     [
#         Input('table-selector', 'value'),
#         Input('score-btn', 'n_clicks'),
#         Input('permanova-btn', 'n_clicks'),
#         Input('anosim-btn', 'n_clicks'),
#         Input('local-t-test-btn', 'n_clicks'),
#         Input("recomputation", "children"),
#         Input("rank-btn",  "n_clicks")
#
#     ],
#     [
#         State("permanova-permutation-nr", "value"),
#         State("anosim-permutation-nr", "value"),
#         State("distance-cutoff", "value"),
#         State('tbl', 'sort_by'),
#         State('data-store', 'data'),
#         State("unique-id", "data"),
#
#     ],
#     prevent_intital_call=True
#
# )
# def new_columns(
#         sel_columns,
#         n_clicks,
#         permanova_clicks,
#         anosim_clicks,
#         t_test_clicks,
#         recompute,
#         ranking,
#         permanova_permutations,
#         anosim_permutations,
#         distance_cutoff,
#         current_sorting,
#         rapdordata,
#         uid
# ):
#     logger.info(f"{ctx.triggered_id} triggered rendering of new table")
#     if rapdordata is None:
#         raise PreventUpdate
#     alert = False
#     run_cluster = dash.no_update
#     sel_columns = [] if sel_columns is None else sel_columns
#     if ctx.triggered_id == "rank-btn":
#         try:
#             cols = [col['column_id'] for col in current_sorting if col != "Rank"]
#             asc = [col['direction'] == "asc" for col in current_sorting if col != "Rank"]
#
#             rapdordata.rank_table(cols, asc)
#             sel_columns += ["Rank"]
#         except Exception as e:
#             alert = True
#             alert_msg = f"Ranking Failed:\n{str(e)}"
#
#     if ctx.triggered_id == "permanova-btn":
#
#         if permanova_clicks == 0:
#             raise PreventUpdate
#         else:
#             sel_columns += ["PERMANOVA F"]
#
#             if permanova_permutations is None:
#                 permanova_permutations = 9999
#             if rapdordata.permutation_sufficient_samples:
#                 rapdordata.calc_permanova_p_value(permutations=permanova_permutations, threads=1, mode="local")
#                 sel_columns += ["local PERMANOVA adj p-Value"]
#
#             else:
#                 rapdordata.calc_permanova_p_value(permutations=permanova_permutations, threads=1, mode="global")
#                 sel_columns += ["global PERMANOVA adj p-Value"]
#
#                 alert = True
#                 alert_msg = "Insufficient Number of Samples per Groups. P-Value is derived using all Proteins as background."
#                 " This might be unreliable"
#     if ctx.triggered_id == "anosim-btn":
#         if anosim_clicks == 0:
#             raise PreventUpdate
#         else:
#             if anosim_permutations is None:
#                 anosim_permutations = 9999
#             sel_columns += ["ANOSIM R"]
#
#             if rapdordata.permutation_sufficient_samples:
#                 rapdordata.calc_anosim_p_value(permutations=anosim_permutations, threads=1, mode="local")
#                 sel_columns += ["local ANOSIM adj p-Value"]
#
#             else:
#                 rapdordata.calc_anosim_p_value(permutations=anosim_permutations, threads=1, mode="global")
#                 sel_columns += ["global ANOSIM adj p-Value"]
#
#                 alert = True
#                 alert_msg = "Insufficient Number of Samples per Groups. P-Value is derived using all Proteins as background."
#                 " This might be unreliable"
#     if ctx.triggered_id == "local-t-test-btn":
#         if "RNase True peak pos" not in rapdordata.df:
#             rapdordata.determine_peaks()
#         rapdordata.calc_welchs_t_test(distance_cutoff=distance_cutoff)
#         sel_columns += ["CTRL Peak adj p-Value", "RNase Peak adj p-Value"]
#
#     if ctx.triggered_id == "score-btn":
#         if n_clicks == 0:
#             raise PreventUpdate
#         else:
#             rapdordata.calc_all_scores()
#             run_cluster = True
#             sel_columns += ["ANOSIM R", "Mean Distance", "shift direction", "RNase False peak pos", "RNase True peak pos", "relative fraction shift"]
#     if alert:
#         alert_msg = html.Div(
#             dbc.Alert(
#                 alert_msg,
#                 color="danger",
#                 dismissable=True,
#             ),
#             className="p-2 align-items-center, alert-msg",
#
#         )
#     else:
#         alert_msg = []
#     #tbl = _create_table(rapdordata, sel_columns)
#     selected_columns = list(set(sel_columns))
#
#     data = rapdordata.extra_df.loc[:, rapdordata._id_columns + selected_columns]
#     columns = []
#     num_cols = ["shift direction"]
#     for i in data.columns:
#         if i != "id":
#             d = dict()
#             d["name"] = str(i)
#             d["id"] = str(i)
#             if is_numeric_dtype(data[i]):
#                 d["type"] = "numeric"
#                 if "p-Value" in i:
#                     d["format"] = Format(precision=2)
#                 else:
#                     d["format"] = Format(precision=4)
#
#                 num_cols.append(str(i))
#             columns.append(d)
#     logger.info(f"Created New Table - sorting: {current_sorting}; run_cluster: {run_cluster}")
#     current_sorting = dash.no_update if current_sorting is None else current_sorting
#     logger.info(data.to_dict("records"))
#     logger.info(columns)
#     return columns, data.to_dict('records'), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


if not DISPLAY:
    @callback(
        Output("table-selector", "value", allow_duplicate=True),
        Output("data-store", "data", allow_duplicate=True),
        Output("run-clustering", "data", allow_duplicate=True),
        Input('score-btn', 'n_clicks'),
        State("table-selector", "value"),
        State("data-store", "data"),
        State("unique-id", "data"),
        background=True,
        running=[
            (Output('score-btn', 'disabled'), True, False),
            (
                    Output("table-progress-container", "style"),
                    {"display": "flex", "height": "90%"},
                    {"display": "none"},
            ),
            (
                    Output("table-loading", "style"),
                    {"display": "none"},
                    {"display": "block", "height": "90%"},
            ),
        ],
        progress=[Output("progress_bar", "value")],

    )
    def run_scoring(set_progress, n_clicks, sel_columns, rapdordata, uid):
        if n_clicks == 0:
            raise PreventUpdate
        else:
            set_progress("50")
            rapdordata.calc_all_scores()
            sel_columns += ["ANOSIM R", "Mean Distance", "position strongest shift"]
            if not rapdordata.categorical_fraction:
                peak_names = rapdordata.score_columns[-2:]
                if isinstance(peak_names, np.ndarray):
                    peak_names = peak_names.tolist()
                sel_columns += ["shift direction", "relative fraction shift"]
                sel_columns += peak_names
            sel_columns = remove_list_duplicates(sel_columns)
            set_progress("100")

        return sel_columns, Serverside(rapdordata, key=uid), True


@callback(
    Output("table-selector", "value", allow_duplicate=True),
    Output("data-store", "data", allow_duplicate=True),
    Output("alert-div", "children", allow_duplicate=True),
    Input('rank-btn', 'n_clicks'),
    State("table-selector", "value"),
    State('tbl', 'sort_by'),

    State("data-store", "data"),
    State("unique-id", "data"),
    prevent_initial_call=True
)
def rank_table(btn, sel_columns, current_sorting, rapdordata, uid):
    alert = False
    if btn is None or btn == 0:
        raise PreventUpdate
    try:
        cols = [col['column_id'] for col in current_sorting if col != "Rank"]
        asc = [col['direction'] == "asc" for col in current_sorting if col != "Rank"]

        rapdordata.rank_table(cols, asc)
        sel_columns += ["Rank"]
        sel_columns = remove_list_duplicates(sel_columns)
    except Exception as e:
        alert = True
        alert_msg = f"Ranking Failed:\n{str(e)}"
    if alert:
        alert_msg = html.Div(
            dbc.Alert(
                alert_msg,
                color="danger",
                dismissable=True,
            ),
            className="p-2 align-items-center, alert-msg",

        )
    else:
        alert_msg = dash.no_update

    return sel_columns, Serverside(rapdordata, key=uid), alert_msg


if not DISPLAY:
    @callback(
        Output("table-selector", "value", allow_duplicate=True),
        Output("data-store", "data", allow_duplicate=True),
        Output("alert-div", "children", allow_duplicate=True),
        Output("tbl", "data", allow_duplicate=True),
        Output("progress_bar", "value"),
        Input('anosim-btn', 'n_clicks'),
        State("table-selector", "value"),
        State("anosim-permutation-nr", "value"),
        State("data-store", "data"),
        State("unique-id", "data"),
        background=True,
        running=[
            (Output('anosim-btn', 'disabled'), True, False),
            (
                    Output("table-progress-container", "style"),
                    {"display": "flex", "height": "90%"},
                    {"display": "none"},
            ),
            (
                    Output("table-loading", "style"),
                    {"display": "none"},
                    {"display": "block", "height": "90%"},
            ),
        ],
        progress=[Output("progress_bar", "value")],
        prevent_initial_call=True

    )
    def run_anosim(set_progress, n_clicks, sel_columns, anosim_permutations, rapdordata, uid):
        alert_msg = dash.no_update
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        else:
            if anosim_permutations is None:
                anosim_permutations = 999
            if anosim_permutations > 9999:
                alert_msg = f"Number of permutations ({anosim_permutations}) too high. " \
                            f"Only less than {MAXPERMUTATIONS} supported."
                alert_msg = html.Div(
                    dbc.Alert(
                        alert_msg,
                        color="danger",
                        dismissable=True,
                    ),
                    className="p-2 align-items-center, alert-msg",

                )
                return dash.no_update, dash.no_update, alert_msg, dash.no_update

            sel_columns += ["ANOSIM R"]

            if rapdordata.permutation_sufficient_samples:
                rapdordata.calc_anosim_p_value(permutations=anosim_permutations, threads=1, mode="local", callback=set_progress)
                sel_columns += ["local ANOSIM adj p-Value"]

            else:
                rapdordata.calc_anosim_p_value(permutations=anosim_permutations, threads=1, mode="global", callback=set_progress)
                sel_columns += ["global ANOSIM adj p-Value"]

                alert_msg = "Insufficient Number of Samples per Groups. P-Value is derived using all Proteins as background."
                " This might be unreliable"
                alert_msg = html.Div(
                    dbc.Alert(
                        alert_msg,
                        color="danger",
                        dismissable=True,
                    ),
                    className="p-2 align-items-center, alert-msg",

                )
        sel_columns = remove_list_duplicates(sel_columns)
        return sel_columns, Serverside(rapdordata, key=uid), alert_msg, dash.no_update, "0"


@callback(
    Output("sel-col-state", "data"),
    Output('table-selector', 'value'),
    Input('table-selector', 'value'),
    State('sel-col-state', 'data'),
    State("data-store", "data"),

)
def set_columns_from_state(selected_columns, sel_col_state, rapdordata):
    logger.info(f"Will update columns: {selected_columns}, col_state: {sel_col_state}")
    sel_col_state = [] if sel_col_state is None else sel_col_state
    check = all(column in rapdordata.extra_df.columns for column in sel_col_state)

    if (selected_columns is None or len(selected_columns) == 0) and len(sel_col_state) > 0 and check:
        selected_columns = sel_col_state
        sel_col_state = dash.no_update
    else:
        sel_col_state = selected_columns
        selected_columns = dash.no_update
    return sel_col_state, selected_columns


@callback(
    Output("tbl", "columns"),
    Output('tbl', 'data', allow_duplicate=True),
    Output("data-table", "style"),
    Input('table-selector', 'value'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def update_columns(selected_columns, rapdordata):

    selected_columns = [] if selected_columns is None else selected_columns

    data = rapdordata.extra_df.loc[0:1, rapdordata._id_columns + selected_columns]
    columns = []
    num_cols = ["shift direction"]
    for i in data.columns:
        if i != "id":
            d = dict()
            d["name"] = str(i)
            d["id"] = str(i)
            if is_numeric_dtype(data[i]):
                d["type"] = "numeric"
                if "p-Value" in i:
                    d["format"] = Format(precision=2)
                else:
                    d["format"] = Format(precision=4)

                num_cols.append(str(i))
            columns.append(d)
    logger.info(f"Updated displayed columns- {columns}")
    style = {"min-width": len(columns) * 120, "overflow-x": "auto"}
    return columns, data.to_dict('records'), style


@callback(
    Output("table-selector", "options", allow_duplicate=True),
    Input('data-store', 'data'),
    Input("table-selector", "options"),

)
def update_selectable_columns(rapdordata, options):
    if rapdordata is None:
        raise PreventUpdate
    new_options = rapdordata.extra_df.columns
    new_options = list(new_options)
    new_options.remove("RAPDORid")
    new_options.remove("id")
    options = dash.no_update if set(new_options) == set(options) else new_options
    return options


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3




operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]
