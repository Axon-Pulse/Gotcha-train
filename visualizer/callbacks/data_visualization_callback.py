import ast
import re

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import callback, callback_context, dcc, html
from dash.dependencies import ALL, MATCH, Input, Output, State
from dash.exceptions import PreventUpdate

import visualizer.layouts.data_visualization

# from visualizer.main_layout import Visualization
# from visualizer.manager import data_handler
import visualizer.main_callbacks

"""
this callback contains the main callback functions for the filter tool.
'update_or_clear_conditions': used to add or clear conditions.
'display_accumulated_conditions': used to present the selected conditions.
'filter_data': used to filter the DataTable based on the stored conditions.
'update_graph': used to display the graph.
'update_histogram': used to display the histogram.
'update_bar_plot': used to display the bar plot.
"""


# data_handler = data_handler()
def register_callbacks(app, visualizer_instance):
    dataset = visualizer_instance.dataset
    df = dataset.dataframe

    # Callback to add a new condition fields:
    @app.callback(
        [Output("conditions-container", "children"), Output("conditions-count", "data")],
        [Input("add-condition-field-btn", "n_clicks")],
        [
            State("filtered-data-table", "data"),
            State("conditions-container", "children"),
            State("conditions-count", "data"),
        ],
    )
    def add_condition_field(n_clicks, filtered_data, existing_conditions, count):
        """
        This function is to add a new condition field.
        when the user clicks on the '+' button, a new condition field will be added.
        """
        existing_conditions = existing_conditions or []
        if n_clicks > 0:
            filtered_data = pd.DataFrame(filtered_data)
            new_condition = dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id={"type": "column-selector", "index": count},
                            options=[
                                {"label": col, "value": col} for col in filtered_data.columns
                            ],
                            style={"width": "60%"},
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dcc.Input(
                            id={"type": "filter-condition", "index": count},
                            type="text",
                            placeholder='Enter text or a numeric range (e.g., "ISAR"/"<200"/"10,20")',
                            style={"width": "80%"},
                        ),
                        width=5,
                    ),
                ],
                style={"margin-bottom": "10px"},
            )  # , id=f'condition-row-{count}')
            existing_conditions.append(new_condition)
            count += 1
        return existing_conditions, count

    # Callback to add or clear conditions:
    @app.callback(
        Output("accumulated-conditions-store", "data"),
        [Input("add-condition-btn", "n_clicks"), Input("clear-filters-btn", "n_clicks")],
        [
            State("accumulated-conditions-store", "data"),
            State({"type": "column-selector", "index": ALL}, "value"),
            State({"type": "filter-condition", "index": ALL}, "value"),
        ],
    )
    def update_or_clear_conditions(
        add_clicks, clear_clicks, existing_conditions, selected_columns, filter_conditions
    ):
        """
        This function is to add or clear conditions to the list of the desired filter.
        'existing_conditions': the conditions that are already added.
        'selected_columns': the columns selected to filter them.
        'filter_conditions': the conditions to filter the selected columns.
        it returns the updated conditions.
        """

        triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
        if triggered_id == "add-condition-btn":
            if not any(selected_columns) or not any(filter_conditions):

                raise PreventUpdate
            existing_conditions = existing_conditions or {}
            for col, cond in zip(selected_columns, filter_conditions):
                if col and cond:
                    existing_conditions[col] = cond
            return existing_conditions
        elif triggered_id == "clear-filters-btn":
            return {}  # , {}   # Clear conditions
        return existing_conditions  # , existing_conditions

    # Callback to present the selected conditions:
    @app.callback(
        Output("conditions-display", "children"), Input("accumulated-conditions-store", "data")
    )
    def display_accumulated_conditions(conditions):
        """
        This function is used to display the selected conditions in the filter tool to the user.
        """

        if conditions is None:
            return html.Ul(html.Li("No conditions added."))
        # Create a list of strings showing each condition:
        conditions_list = [
            f"Condition {list(conditions.keys()).index(column)+1} is: '{column}': '{cond}'"
            for column, cond in conditions.items()
        ]
        return html.Ul([html.Li(cond_str) for cond_str in conditions_list])

    # Callback to filter the DataTable based on the stored conditions:
    @app.callback(
        [
            Output("filtered-data-table", "data"),
            Output("filtered_data_store", "data"),
            Output("filter-error-message", "style"),
        ],
        [Input("filter-table-btn", "n_clicks"), Input("clear-filters-btn", "n_clicks")],
        State("accumulated-conditions-store", "data"),
        State("filtered_data_store", "data"),
    )
    def filter_data(filter_btn, n_clicks2, existing_conditions, filtered_data):
        """
        This function is used to filter the DataTable based on the stored conditions.
        'existing_conditions': the conditions that are already added.
        'filtered_data': the data after filtering.
        'apply_conditions': the function to apply the conditions on the DataTable.
        """
        triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
        if filter_btn is None:
            raise PreventUpdate
        else:  # Filter or clear the data based on the selected column and condition
            if triggered_id == "clear-filters-btn":
                filtered_data = df
                error_message = {"color": "red", "display": "none"}
                return (
                    filtered_data.to_dict("records"),
                    filtered_data.to_dict("records"),
                    error_message,
                )
            elif triggered_id == "filter-table-btn":
                print("existing_conditions:", existing_conditions)
                unfiltered_data = pd.DataFrame(filtered_data)
                filtered_data = unfiltered_data
                for col, cond in existing_conditions.items():
                    col_type = filtered_data[col].dtype
                    print("col:", col)
                    print("cond:", cond)
                    try:
                        filtered_data = apply_conditions(col, cond, col_type, filtered_data)
                        error_message = {"color": "red", "display": "none"}
                    except ValueError as e:
                        filtered_data = {}
                        error_message = {"color": "red", "display": "block", "message": str(e)}
                    except Exception as e:
                        filtered_data = {}
                        error_message = {"color": "red", "display": "block", "message": str(e)}

                if isinstance(filtered_data, pd.DataFrame):
                    return (
                        filtered_data.to_dict("records"),
                        filtered_data.to_dict("records"),
                        error_message,
                    )
                else:
                    return filtered_data, filtered_data, error_message
            else:
                raise PreventUpdate

    def apply_conditions(col, cond, col_type, data):
        """
        This function is used to apply the conditions on the DataTable.
        'col': the column to apply the condition on.
        'cond': the condition to apply on the column.
        'col_type': the type of the column.
        'data': the data to apply the conditions on.
        'filtered_data': the data after filtering.
        """

        operator_map = {
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
            ">": lambda x, y: x > y,
            "<": lambda x, y: x < y,
            "=": lambda x, y: x == y,
            "!=": lambda x, y: x != y,
        }
        if pd.api.types.is_string_dtype(col_type):
            # Handle string search
            if re.match(r"^[<>!=]", cond) or "," in cond or re.match(r"^[0-9]", cond):
                print(cond, col_type)
                raise ValueError(f"Condition '{cond}' is not valid for string column '{col}'")
            filtered_data = data[data[col].str.contains(cond, case=False, na=False)]
        elif pd.api.types.is_numeric_dtype(col_type):
            # Handle numeric range. Expecting filter_condition to be "start,end"
            if "," in cond:
                try:
                    start, end = map(float, cond.split(","))
                except ValueError:
                    raise ValueError(
                        f"Condition '{cond}' is not a valid numeric range for column '{col}'"
                    )
                filtered_data = data[data[col].between(start, end)]
            else:
                operator_pattern = re.compile(r"([<>]=?|!=|=)")
                match = operator_pattern.search(cond)
                if match:
                    operator = match.group(0)
                    try:
                        value = float(cond.split(operator)[1])
                        filtered_data = data[operator_map[operator](data[col], value)]
                    except ValueError:
                        raise ValueError(
                            f"Condition '{cond}' is not a valid numeric condition for column '{col}'"
                        )
                else:
                    try:
                        single_value = float(cond)
                        filtered_data = data[data[col] == single_value]
                        print("single_value:", single_value)
                        print("filtered_data:", filtered_data.shape)
                    except ValueError:
                        raise ValueError(
                            f"Condition '{cond}' is not a valid numeric condition for column '{col}'"
                        )
        return filtered_data


# The callbacks for the graphs of the filtered data located at 'data_visualization_2_callback.py'
