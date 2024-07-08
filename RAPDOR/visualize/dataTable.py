import dash_loading_spinners as dls
import pandas as pd
from dash import html, dcc, dash_table
from dash.dash_table.Format import Format
from pandas.core.dtypes.common import is_numeric_dtype
import dash_bootstrap_components as dbc
import logging

logger = logging.getLogger(__name__)


def _get_table(rapdordata):
    if rapdordata is not None:
        sel_columns = []
        for name in rapdordata.score_columns:
            if name in rapdordata.extra_df:
                sel_columns.append(name)
        logger.info(f"Initial Columns: {sel_columns}")
    else:
        sel_columns = None
    table = html.Div(
        [
            html.Div(
                html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        _create_table(rapdordata),
                                        className="col-12 justify-content-center h-100 dbc-row-selectable",
                                        id="data-table",
                                        style={"min-width": "100px", "overflow-x": "auto",
                                               # "display": "block"
                                               }

                                    ),

                                    html.Button("Reset Selected Rows", id="reset-rows-btn",
                                                className="reset-col-btn first-page",
                                                style=dict(position="absolute", left="0")),

                                ],

                                id="table-loading",
                                style={"height": "90%"}

                            ),


                            html.Div(
                                html.Div(
                                    dbc.Progress(
                                        id="progress_bar",
                                        value="0",
                                        color="var(--primary-color)",
                                        animated=True,
                                        striped=True,
                                        style={"padding": "0"}),
                                    className="w-80",
                                    style={"height": "1rem", "width": "90%"}

                                ),

                                className="col-12 justify-content-center align-items-center p-2",
                                style={"display": "none"},
                                id="table-progress-container"

                            ),


                            html.Div(
                                dcc.Dropdown(
                                    [sel_columns] if sel_columns is not None else [],
                                    placeholder="Select Table Columns",
                                    className="justify-content-center dropUp",
                                    multi=True,
                                    id="table-selector",
                                    persistence=True,
                                    persistence_type="session"
                                ),
                                className="col-12 pt-1",
                                id="tbl-dropdown"
                            ),

                        ],


                    className="row justify-content-center h-100"
                ),

                className="databox databox-open p-3", style={"resize": "vertical", "overflow-y": "auto", "min-height": "470px", "height": "470px"}, id="table-tut"
            )
        ],
        className="col-12 px-1 pb-1 justify-content-center",
    )
    return table


def _create_table(rapdordata, selected_columns=None):
    if selected_columns is None:
        selected_columns = []
    selected_columns = list(set(selected_columns))

    if rapdordata is None:
        columns = [{'name': 'RAPDORid', 'id': 'RAPDORid', 'type': 'numeric'}]
        data = [{"empty": 1, "Empty2": 2}]
        return html.Div(dash_table.DataTable(
            data,
            columns,
            id="tbl",
            sort_action="custom",
            sort_mode="multi",
            sort_by=[],
            row_selectable="multi",
            filter_action='custom',
            filter_query='',
            page_size=50,
            page_current=0,
            page_action="custom",
            style_table={'overflowX': 'auto', "padding": "1px", "height": "300px",
                         "overflowY": "auto"},
            fixed_rows={'headers': True},
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'color': 'white',
                "border": "1px",
                "font-family": "var(--bs-body-font-family)"

            },
            style_data={
                'color': 'var(--r-text-color)',
                "border": "1px",
                "font-family": "var(--bs-body-font-family)",
                "cursor": "pointer"

            },
            style_data_conditional=SELECTED_STYLE,
            style_cell={
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'maxWidth': 0
            },
            style_filter={
                "color": "white",
                "border-color": "red"
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'RAPDORid'},
                    'textAlign': 'left',

                }
            ]
        ), className=" h-100"
        )

    data = rapdordata.extra_df.loc[:, rapdordata._id_columns + selected_columns]
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
    width = "10%" if len(columns) > 1 else "98%"
    t = html.Div(dash_table.DataTable(
        data.to_dict('records'),
        columns,
        id='tbl',
        sort_action="custom",
        sort_mode="multi",
        sort_by=[],
        row_selectable="multi",

        filter_action='custom',
        filter_query='',
        page_size=50,
        page_current=0,
        page_action="custom",
        style_table={'overflowX': 'auto', "padding": "1px", "height": "300px",
                     "overflowY": "auto"},
        fixed_rows={'headers': True},
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white',
            "border": "1px",
            "font-family": "var(--bs-body-font-family)"

        },
        style_data={
            'color': 'var(--r-text-color)',
            "border": "1px",
            "font-family": "var(--bs-body-font-family)"

        },
        style_data_conditional=SELECTED_STYLE,
        style_cell={
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'maxWidth': 0
        },
        style_filter={
            "color": "white",
            "border-color": "red"
        },
        style_cell_conditional=[
                                   {
                                       'if': {'column_id': 'RAPDORid'},
                                       'textAlign': 'left',
                                       "width": width
                                   }
                               ]
    ), className="h-100"
    )

    return t


SELECTED_STYLE = [
        {
            "if": {"state": "active"},
            "border-top": "0px solid var(--primary-color)",
            "border-bottom": "0px solid var(--primary-color)",
            "border-left": "0px solid var(--primary-color)",
            "border-right": "0px solid var(--primary-color)",
        },
        {
            "if": {"state": "selected"},
            "border-top": "0px solid var(--primary-color)",
            "border-bottom": "0px solid var(--primary-color)",
            "border-left": "0px solid var(--primary-color)",
            "border-right": "0px solid var(--primary-color)",
        },
    ]
