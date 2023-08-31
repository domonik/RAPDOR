import dash_loading_spinners as dls
import pandas as pd
from dash import html, dcc, dash_table
from dash.dash_table.Format import Format
from pandas.core.dtypes.common import is_numeric_dtype
import logging

logger = logging.getLogger(__name__)


def _get_table(rdpmsdata):
    table = html.Div(
        [
            html.Div(
                html.Div(
                        [

                            dls.RingChase(
                                html.Div(
                                    _create_table(rdpmsdata),
                                    className="col-12 justify-content-center h-100 dbc-row-selectable",
                                    id="data-table"

                                ),
                                color="var(--primary-color)",
                                width=200,
                                thickness=20,
                                id="ring-chase-tbl"

                            ),
                            html.Div(
                                dcc.Dropdown(
                                    [],
                                    placeholder="Select Table Columns",
                                    className="justify-content-center dropUp",
                                    multi=True,
                                    id="table-selector"
                                ),
                                className="col-12 pt-1",
                                id="tbl-dropdown"
                            ),

                        ],


                    className="row justify-content-center h-100"
                ),

                className="databox databox-open p-3", style={"resize": "vertical", "overflow-y": "auto", "min-height": "470px", "height": "470px"}
            )
        ],
        className="col-12 px-1 pb-1 justify-content-center",
    )
    return table


def _create_table(rdpmsdata, selected_columns=None):
    if selected_columns is None:
        selected_columns = []
    if rdpmsdata is None:
        columns = ["empty", "Empty2"]
        data = [{"empty": 1, "Empty2": 2}]
        return html.Div(dash_table.DataTable(
            data,
            id="tbl",
        ), className="dont-show h-100"
        )

    data = rdpmsdata.extra_df.loc[:, rdpmsdata._id_columns + selected_columns]
    for name in rdpmsdata.score_columns:
        if name in rdpmsdata.extra_df:
            data = pd.concat((data, rdpmsdata.extra_df[name]), axis=1)
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
                                       'if': {'column_id': 'RDPMSpecID'},
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
