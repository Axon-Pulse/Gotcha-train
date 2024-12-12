import os

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html
from flask import Flask

import visualizer.callbacks.add_layout_callback as callback

#  need to add information about the additions to the new files

# Create directories if they don't exist

# Flask server for deployment
server = Flask(__name__)


def add_layout_layout():
    """
    This function is the layout for the 'add layout' page.
    """
    layout = dbc.Container(
        [
            html.H2("Add Layout", style={"textAlign": "center"}),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Markdown(
                                """
            ### Instructions:
            To integrate your uploaded scripts into the visualizer, follow these steps:

            1. **Upload your layout script**. It will be saved in the `layout/` directory.
            2. **Upload your callback script**. It will be saved in the `callbacks/` directory.

            4. If you want to use objects from the main layout of this visualizer, ensure that in your layout and callback scripts you import the following:
               ```python
                import main_layout
                import main_callbacks
                ```
            5. Ensure that in your **callback** script you import the layout script:
                ```python
                import visualizer.layouts.<layout_script_name>
               ```
            6. Ensure that in your **layout** script you import the callback script:
                ```python
                import visualizer.callback.<callback_script_name>
                ```
            7. For more convenience, in the layout or callback script, create an instance of the Visualization class:
                ```python
                visualizer_instance = main_layout.Visualization()
                ```
            8. **In `main_layout.py` inside `create_layout` function, add a condition for the new layout:**
               ```python
               elif '<new_layout_name>' in selected_option:
                   import visualizer.layouts.<new_layout_name> as <new_layout_name>
                   layout = <new_layout_name>.<new_layout_name>_layout()
               ```
            9. **In `main_layout.py` inside 'default_layout' function, add new options to the dropdown menu:**
               ```python
               {'label': '<new_option>', 'value': '<new_option>'}
               ```
        """
                            )
                        ],
                        width=12,
                        style={"margin-bottom": "20px"},
                    ),
                    dbc.Row([dbc.Col(html.H1("User-Defined Tool Upload"), width=10)]),
                    dbc.Row([dbc.Col(html.H4("Upload the Layout Script:"), width=8)]),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Upload(
                                    id="upload-layout",
                                    children=html.Div(
                                        ["Drag and Drop or ", html.A("Select Files")]
                                    ),
                                    style={
                                        "width": "100%",
                                        "height": "40px",
                                        "lineHeight": "40px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "10px",
                                    },
                                    multiple=False,
                                ),
                                width=6,
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    id="layout-upload-name",
                                    style={"marginTop": "10px", "color": "green"},
                                ),
                                width=12,
                            )
                        ]
                    ),
                    dbc.Row([dbc.Col(html.H4("Upload the Callback Script:"), width=8)]),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Upload(
                                    id="upload-callback",
                                    children=html.Div(
                                        ["Drag and Drop or ", html.A("Select Files")]
                                    ),
                                    style={
                                        "width": "100%",
                                        "height": "40px",
                                        "lineHeight": "40px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "10px",
                                    },
                                    multiple=False,
                                ),
                                width=6,
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    id="callback-upload-name",
                                    style={"marginTop": "10px", "color": "green"},
                                ),
                                width=12,
                            )
                        ]
                    ),
                    dbc.Row(
                        [dbc.Col(dbc.Button("Upload", id="upload-button", n_clicks=0), width=4)]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Loading(
                                    id="loading",
                                    children=[html.Div(id="output-status")],
                                    type="default",
                                ),
                                width=12,
                            )
                        ]
                    ),
                ]
            ),
        ],
        fluid=True,
    )
    return layout
