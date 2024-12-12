import base64
import io

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, Input, Output, callback_context, dash_table, dcc, html
from dash.dependencies import ALL, Input, Output, State
from dash.exceptions import PreventUpdate
from PIL import Image

import visualizer.callbacks.data_visualization_2_callback
import visualizer.callbacks.data_visualization_callback
import visualizer.main_callbacks
import visualizer.main_layout

"""
this is the layout function of the data analysis page. it contains filter tool and the data graphs
(column vs column, histogram, bar plot).
"""


def data_visualization_layout(dataset):
    df = dataset.dataframe
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Data Visualization", style={"textAlign": "center"}),
                    dbc.Container(
                        [
                            html.Hr(),
                            html.H3("Filter Tool"),
                            dbc.Row(
                                [
                                    dbc.Col(html.Label("Select Column for Filtering:"), width=5),
                                    dbc.Col(html.Label("Enter Filter Condition:"), width=5),
                                    dbc.Col(
                                        dbc.Button(
                                            "+",
                                            id="add-condition-field-btn",
                                            n_clicks=0,
                                            color="primary",
                                        ),
                                        width=1,
                                    ),
                                ]
                            ),
                            html.Hr(),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id={"type": "column-selector", "index": 0},
                                            options=[
                                                {"label": col, "value": col} for col in df.columns
                                            ],
                                            style={"width": "100%"},
                                        ),
                                        width=5,
                                    ),
                                    dbc.Col(
                                        dcc.Input(
                                            id={"type": "filter-condition", "index": 0},
                                            type="text",
                                            placeholder='Enter text or a numeric range (e.g., "ISAR"/"<200"/"10,20")',
                                            style={"width": "100%"},
                                        ),
                                        width=6,
                                    ),
                                ],
                                style={"margin-bottom": "10px"},
                            ),  # , id=f'condition-row-{count}')
                            html.Div(id="conditions-container"),
                        ]
                    ),
                    dbc.Button(
                        "List Filter Conditions",
                        id="add-condition-btn",
                        n_clicks=0,
                        style={"marginRight": "10px"},
                    ),
                    html.Div(id="conditions-display", className="mb-3"),
                    html.Div(
                        "Error processing filter condition. Please check your input format.",
                        id="filter-error-message",
                        style={"color": "red", "display": "none"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    id="buttons",
                                    children=[
                                        dbc.Button(
                                            "Clear Filters",
                                            id="clear-filters-btn",
                                            n_clicks=0,
                                            style={"backgroundColor": "#FF4136", "color": "white"},
                                        ),
                                        dbc.Button(
                                            "Filter Table",
                                            id="filter-table-btn",
                                            n_clicks=0,
                                            style={
                                                "backgroundColor": "#28a745",
                                                "color": "white",
                                                "marginLeft": "10px",
                                            },
                                        ),
                                    ],
                                )
                            ),
                        ]
                    ),
                    html.Hr(),
                    html.H3("Data Graphs"),
                    html.Label("Select Number of graphs you want:"),
                    dbc.Row(
                        dcc.Dropdown(
                            id="num-graphs-dropdown",
                            options=[{"label": i, "value": i} for i in range(1, 10)],
                            style={"width": "100%", "marginBottom": "5px"},
                        ),
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button("Export Graphs", id="export-button"),
                                width=3,
                                style={"marginRight": "5px"},
                            ),
                            dbc.Col(
                                dcc.Upload(
                                    id="import-upload",
                                    children=dbc.Button("Import Graph", id="import-btn"),
                                    multiple=False,
                                    style={"width": "100%"},
                                ),
                                width=3,
                                style={"marginRight": "5px"},
                            ),
                            dbc.Col(
                                dbc.Button("Show Graphs", id="show-graph", n_clicks=0), width=3
                            ),
                        ]
                    ),
                ],
                className="sidebar",
            ),
            html.Div(
                [
                    html.H2("Data Filter", style={"textAlign": "center"}),
                    dcc.Markdown(
                        """
                ##### Here you can filter your data based on the condition you want. In order to filter the data, follow thw instructions:

                1. Select the **column** you want to enter your conditions.
                2. write the **condition** you want to apply to the column. the conditions need to be in the following format:
                    - If you want to perform the filtering according to a certain threshold value, write: `<200` (supported operators: `'<,>,=,>=,<=,!='`).
                    - If you want to filter by a range of values, write: `10,20`. (for a single value you can write only: `200`.)
                    - If you want to filter by a string value write: `ISAR`.
                3. You can add multiple conditions using **'+'** button.
                4. Click on **'List Filter Conditions'**. This button will display the columns you selected with the conditions you filled in, provided they match the `dtype` of values in the column.
                5. Click on **'Filter Table'** to view the filtered data in the table. and Click **'Clear Filters'** to remove the conditions and view the table without conditions.
                """
                    ),
                    dash_table.DataTable(
                        id="filtered-data-table",
                        columns=[{"name": i, "id": i} for i in df.columns],
                        data=df.to_dict("records"),
                        sort_action="native",  # Enable sorting
                        sort_mode="multi",  # Allow multi-column sorting
                        row_selectable="single",  # Allow single row selection (for simplicity)
                        page_action="native",  # Enable pagination
                        page_current=0,
                        page_size=20,  # Number of rows per page
                        style_table={"height": "400px", "overflowY": "auto"},
                        style_header={
                            "backgroundColor": "rgb(230, 230, 230)",
                            "fontWeight": "bold",
                        },
                        style_cell={"textAlign": "left"},
                    ),
                    html.Hr(),
                    html.H2("Data Graphs", style={"textAlign": "center"}),
                    dcc.Markdown(
                        """
            #### Here you can visualize your data in the form of graphs.
            ###### The following steps will help you to design your graphs:
            1. You can select the number of graphs you want to display and the type of graph you want to display.

                **Notice:** In order not to have to repeat the processing of the graphs from the beginning, **First** you need to select the number of graphs you want to display, then select the type of graph you want to display.

            2. You can export the graphs you designed as `.json` file, by click on **'Export Graphs'**.

            3. If you want to review the exported graphs, you can import them by clicking **'Import Graphs'**, select the exported file, and then click on **'Show Graphs'**.

                **Notice:** if you chose to design new graphs, you cannot import graphs at the same time, but only 'overwrite' the new graphs!.
                And also, if you choose to import graphs you can't design new graphs **until you end the import process** (clicking on 'Show Graphs').


            4. The graph's data is based on the filtered data from the table above. If you change the data in the table and you want view the changes
                in the graph, you need to click 'Generate Graph' in order to see the updated changes.
            """
                    ),
                    dcc.Store(
                        id="accumulated-conditions-store"
                    ),  # Stores the accumulated conditions
                    dcc.Store(id="filtered_data_store", data=df.to_dict("records")),
                    dcc.Store(id="conditions-count", data=1),
                    # ]),
                    # dbc.Button("Save as CSV", id="save-csv-btn", n_clicks=0, style={'backgroundColor': '#28a745', 'color': 'white'}),
                    # dcc.Download(id='download-link'),
                    html.Hr(),
                    html.Div(
                        id="graph-container",
                        style={
                            "display": "flex",
                            "flexWrap": "wrap",
                            "justifyContent": "flex-start",
                        },
                        children=[],
                    ),
                    dcc.Store(id="full data", data={}),
                    dcc.Download(id="download-graphs"),
                ],
                className="content",
            ),
        ]
    )
