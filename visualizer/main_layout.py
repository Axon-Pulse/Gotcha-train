import base64
import io

import dash_bootstrap_components as dbc
import hydra
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import torch
from dash import Dash, Input, Output, callback_context, dash_table, dcc, html
from dash.dependencies import ALL, Input, Output, State
from dash.exceptions import PreventUpdate
from lightning import LightningDataModule
from omegaconf import DictConfig
from PIL import Image

"""
this file is used to display the main layout: the default layout of the page and the layout based on the selected option.
it also manage the layouts and callbacks of the page based on the selected option in the dropown menu.
'default_layout': used to display the default layout of the page (the 'dropdown menu'- the features of the visualizer,
                    and the 'data table'- the meta data of the project).
'create_layout': used to create the layout based on the selected option.
in order to add more features to the visualizer, you can add more options to the dropdown menu, insise the 'options' list:
{'label': new option, 'value': new_option},

"""


class Visualization:

    def __init__(
        self,
        dataset: torch.utils.data.Dataset = None,
        model=None,
        transforms: DictConfig = None,
        default_transforms: DictConfig = None,
    ):

        self.dataset = dataset
        self.model = model
        self.transforms = transforms
        self.default_transforms = default_transforms

        # flags for interactive connection with the visualizer:
        self.imports = False  # ACK flag for importing the layout and callback scripts
        self.flag = False  # ACK flag for preventing the callback from repeating
        self.trans_dict = None  # flag for save the transforms values
        self.vis_type = "img"  # flag for the type of the visualization (images or graphs)
        self.layout = self.default_layout()

    def default_layout(self):
        table = self.dataset.dataframe
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(width=3),
                        dbc.Col(
                            dcc.Dropdown(
                                id="display-option-selector",
                                options=[
                                    {"label": "Data Analysis", "value": "data_analysis"},
                                    {"label": "Data Visualization", "value": "data_visualization"},
                                    {"label": "Add new layout", "value": "add_layout"},
                                ],
                                value="data_analysis",
                            ),
                            width=3,
                        ),
                    ],
                    className="mb-3",
                ),
                html.Div(
                    [
                        dbc.Container(
                            [
                                dash_table.DataTable(
                                    data=table.to_dict("records"),
                                    columns=[{"name": i, "id": i} for i in table.columns],
                                    sort_action="native",  # Enable sorting
                                    sort_mode="multi",  # Allow multi-column sorting
                                    row_selectable="multi",  # Allow single row selection (for simplicity)
                                    page_action="native",  # Enable pagination
                                    page_current=0,
                                    page_size=20,  # Number of rows per page
                                    style_table={
                                        "height": "400px",
                                        "overflowY": "auto",
                                        "display": "block",
                                    },
                                    style_header={
                                        "backgroundColor": "rgb(230, 230, 230)",
                                        "fontWeight": "bold",
                                    },
                                    style_cell={"textAlign": "left"},
                                    id="table-data",
                                ),
                                dbc.Button(
                                    "View Data Table",
                                    id="view-table-btn",
                                    className="mb-3",
                                    style={"backgroundColor": "#007BFF"},
                                ),
                                dbc.Button(
                                    "Hide Data Table",
                                    id="hide-table-btn",
                                    className="mb-3",
                                    style={"backgroundColor": "#FF4136", "margin-left": "10px"},
                                ),
                                # dbc.Button("Export Config file", id="config-btn", className="mb-3",style={'backgroundColor': '#28a745', 'margin-left': '10px'}),
                            ],
                            id="table-container",
                            style={"display": "block"},
                        ),
                    ]
                ),
            ],
            className="content",
        )

    # Create the layoutr based on the selected option
    def create_layout(self, selected_option):
        if "data_analysis" in selected_option:
            from visualizer.layouts.data_analysis import DataAnalysisLayout

            layout = DataAnalysisLayout().data_analysis_layout(
                self.transforms, self.default_transforms
            )
        elif "data_visualization" in selected_option:
            import visualizer.layouts.data_visualization as data_visualization

            layout = data_visualization.data_visualization_layout(self.dataset)
        elif "add_layout" in selected_option:
            import visualizer.layouts.add_layout as add_layout

            layout = add_layout.add_layout_layout()
        return layout
