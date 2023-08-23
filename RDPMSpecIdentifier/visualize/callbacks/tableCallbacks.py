import os

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Output, Input, State, ctx, html
from dash.exceptions import PreventUpdate

from RDPMSpecIdentifier.visualize.appDefinition import app
from RDPMSpecIdentifier.visualize.dataTable import SELECTED_STYLE, _create_table
from dash_extensions.enrich import Serverside, State


@app.callback(
    Output('tbl', 'data'),
    Output('tbl', "page_current"),
    Output("tbl-store", "data"),

    Input('tbl', "page_current"),
    Input('tbl', "page_size"),
    Input('tbl', 'sort_by'),
    Input('tbl', 'filter_query'),
    State('table-selector', 'value'),
    State("protein-id", "children"),
    State("data-store", "data"),
    State("unique-id", "data"),

)
def update_table(page_current, page_size, sort_by, filter, selected_columns, key, rdpmspec, uid):
    key = key.split("Protein ")[-1]
    if selected_columns is None:
        selected_columns = []

    data = rdpmspec.extra_df.loc[:, rdpmspec._id_columns + selected_columns]
    for name in rdpmspec.score_columns:
        if name in rdpmspec.extra_df:
            data = pd.concat((data, rdpmspec.extra_df[name]), axis=1)

    filtering_expressions = filter.split(' && ')
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            data = data.loc[getattr(data[col_name], operator)(filter_value)]
        elif operator == 'contains':
            filter_value = str(filter_value).split(".0")[0]
            data = data.loc[data[col_name].str.contains(filter_value).fillna(False)]
        elif operator == 'datestartswith':
            filter_value = str(filter_value).split(".0")[0]

            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            data = data.loc[data[col_name].str.startswith(filter_value)]

    if sort_by is not None:
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
        size = page_size
    elif key in data.index:
        loc = data.index.get_loc(key)
        page = int(np.floor(loc / page_size))
        size = page_size
    else:
        page = page_current
        size = page_size
    return data.iloc[page * size: (page + 1) * size].to_dict('records'), page, Serverside(data, uid + "tbl")


@app.callback(
    Output("tbl", "style_data_conditional"),
    Input('tbl', 'active_cell'),
    Input('tbl', 'data'),
    State("protein-id", "children"),
    State("tbl", "page_size"),
    State('tbl', "page_current"),
    State("tbl-store", "data"),

)
def style_selected_col(active_cell, sort_by, key, page_size, current_page, data):
    if "tbl.active_cell" in ctx.triggered_prop_ids:
        if active_cell is None:
            raise PreventUpdate
    if "tbl.data" in ctx.triggered_prop_ids:
        key = key.split("Protein ")[-1]
        if key in data.index:
            loc = data.index.get_loc(key)
            page = int(np.floor(loc / page_size))
            if page != current_page:
                row_idx = -1
            else:
                row_idx = int(loc % page_size)
        else:
            row_idx = -1
    else:
        if active_cell is None:
            raise PreventUpdate
        row_idx = active_cell["row"]

    style = [
        {
            "if": {"row_index": row_idx},
            "backgroundColor": "red !important",
            "border-top": "2px solid var(--primary-color)",
            "border-bottom": "2px solid var(--primary-color)",
            "border-left": "0px solid var(--primary-color)",
            "border-right": "0px solid var(--primary-color)",
        },
    ]
    style_data_conditional = SELECTED_STYLE + style
    return style_data_conditional


@app.callback(
    [
        Output("data-table", "children"),
        Output("alert-div", "children", allow_duplicate=True),
        Output('tbl', 'sort_by'),
        Output('data-store', 'data', allow_duplicate=True),
    ],
    [
        Input('table-selector', 'value'),
        Input('score-btn', 'n_clicks'),
        Input('permanova-btn', 'n_clicks'),
        Input('anosim-btn', 'n_clicks'),
        Input('local-t-test-btn', 'n_clicks'),
        Input("recomputation", "children"),
        Input("rank-btn",  "n_clicks")

    ],
    [
        State("permanova-permutation-nr", "value"),
        State("anosim-permutation-nr", "value"),
        State("distance-cutoff", "value"),
        State('tbl', 'sort_by'),
        State('data-store', 'data'),
        State("unique-id", "data"),

    ],
    prevent_intital_call="initial_duplicate"

)
def new_columns(
        sel_columns,
        n_clicks,
        permanova_clicks,
        anosim_clicks,
        t_test_clicks,
        recompute,
        ranking,
        permanova_permutations,
        anosim_permutations,
        distance_cutoff,
        current_sorting,
        rdpmsdata,
        uid
):
    alert = False
    if ctx.triggered_id == "rank-btn":
        try:
            cols = [col['column_id'] for col in current_sorting if col != "Rank"]
            asc = [col['direction'] == "asc" for col in current_sorting if col != "Rank"]

            rdpmsdata.rank_table(cols, asc)
        except Exception as e:
            alert = True
            alert_msg = f"Ranking Failed:\n{str(e)}"

    if ctx.triggered_id == "permanova-btn":

        if permanova_clicks == 0:
            raise PreventUpdate
        else:
            if permanova_permutations is None:
                permanova_permutations = 9999
            if rdpmsdata.permutation_sufficient_samples:
                rdpmsdata.calc_permanova_p_value(permutations=permanova_permutations, threads=os.cpu_count(), mode="local")
            else:
                rdpmsdata.calc_permanova_p_value(permutations=permanova_permutations, threads=os.cpu_count(), mode="global")

                alert = True
                alert_msg = "Insufficient Number of Samples per Groups. P-Value is derived using all Proteins as background."
                " This might be unreliable"
    if ctx.triggered_id == "anosim-btn":
        if anosim_clicks == 0:
            raise PreventUpdate
        else:
            if anosim_permutations is None:
                anosim_permutations = 9999
            if rdpmsdata.permutation_sufficient_samples:
                rdpmsdata.calc_anosim_p_value(permutations=anosim_permutations, threads=os.cpu_count(), mode="local")
            else:
                rdpmsdata.calc_anosim_p_value(permutations=anosim_permutations, threads=os.cpu_count(), mode="global")
                alert = True
                alert_msg = "Insufficient Number of Samples per Groups. P-Value is derived using all Proteins as background."
                " This might be unreliable"
    if ctx.triggered_id == "local-t-test-btn":
        if "RNAse True peak pos" not in rdpmsdata.df:
            rdpmsdata.determine_peaks()
        rdpmsdata.calc_welchs_t_test(distance_cutoff=distance_cutoff)

    if ctx.triggered_id == "score-btn":
        if n_clicks == 0:
            raise PreventUpdate
        else:
            rdpmsdata.calc_all_scores()
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
        alert_msg = []

    return _create_table(rdpmsdata, sel_columns), alert_msg, current_sorting, Serverside(rdpmsdata, key=uid)


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


@app.callback(
    Output("tbl", "active_cell"),
    Input('tbl', 'selected_row_ids'),
)
def reset_selection(selected_row_ids):
    if selected_row_ids is None or len(selected_row_ids) == 0:
        raise PreventUpdate
    return None


operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]