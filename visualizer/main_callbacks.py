import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate

# import main_layout
"""
this file contains the main callback functions for the visualizer app.
'display_table': used to display or hide the data table based on the selected option.
'display_page': used to display the page based on the selected option.
"""


def visualizer_callbacks(app, visualizer_instance):
    import visualizer.callbacks.add_layout_callback as add_layout_callback
    import visualizer.callbacks.data_analysis_callback as data_analysis_callback
    import visualizer.callbacks.data_visualization_2_callback as data_visualization_2_callback
    import visualizer.callbacks.data_visualization_callback as data_visualization_callback

    data_analysis_callback.register_callbacks(app, visualizer_instance)
    data_visualization_callback.register_callbacks(app, visualizer_instance)
    data_visualization_2_callback.register_callbacks(app, visualizer_instance)
    add_layout_callback.register_callbacks(app, visualizer_instance)

    @app.callback(
        Output("table-data", "style_table"),
        [Input("view-table-btn", "n_clicks"), Input("hide-table-btn", "n_clicks")],
    )
    def _display_table(view_table, hide_table):
        triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
        if triggered_id == "hide-table-btn":
            return {"display": "none"}
        elif triggered_id == "view-table-btn" or not hide_table:
            return {"display": "block"}

    @app.callback(
        Output("table-container", "style"),
        Input("display-option-selector", "value"),
    )
    def _display_table(selected_option):
        if selected_option == "data_analysis":
            return {"display": "block"}
        else:
            return {"display": "none"}

    @app.callback(Output("page-content", "children"), Input("display-option-selector", "value"))
    def _display_page(selected_option):
        return html.Div(
            id="app-container",
            children=[visualizer_instance.create_layout(selected_option)],
        )
