import base64
import os

from dash import Dash, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

import visualizer.layouts.add_layout
import visualizer.main_callbacks
import visualizer.main_layout

#  need to add information about the additions to the new files
os.makedirs(os.getcwd() + "/visualizer/layouts", exist_ok=True)
os.makedirs(os.getcwd() + "/visualizer/callbacks", exist_ok=True)


def save_file(name, content, folder):
    content_type, content_string = content.split(",")
    decoded = base64.b64decode(content_string)
    file_path = os.path.join(folder, name)
    with open(file_path, "wb") as f:
        f.write(decoded)


def register_callbacks(app, visualizer_instance):
    # Callback to handle file uploads and display file names
    @app.callback(Output("layout-upload-name", "children"), [Input("upload-layout", "filename")])
    def update_layout_upload_name(filename):
        if not filename:
            raise PreventUpdate
        else:
            return f"Uploaded file: {filename}"

    @app.callback(
        Output("callback-upload-name", "children"), [Input("upload-callback", "filename")]
    )
    def update_callback_upload_name(filename):
        if not filename:
            raise PreventUpdate
        else:
            return f"Uploaded file: {filename}"

    @app.callback(
        Output("output-status", "children"),
        [Input("upload-button", "n_clicks")],
        [
            State("upload-layout", "filename"),
            State("upload-layout", "contents"),
            State("upload-callback", "filename"),
            State("upload-callback", "contents"),
        ],
    )
    def update_output(
        n_clicks, layout_filename, layout_content, callback_filename, callback_content
    ):
        if n_clicks == 0:
            raise PreventUpdate

        status_messages = []

        if layout_filename and layout_content:
            save_file(layout_filename, layout_content, os.getcwd() + "/visualizer/layouts")
            status_messages.append(
                html.Div(
                    f"Layout script {layout_filename} saved to 'visualizer/layout/' directory.",
                    style={"color": "green"},
                )
            )

        if callback_filename and callback_content:
            save_file(callback_filename, callback_content, os.getcwd() + "/visualizer/callbacks")
            status_messages.append(
                html.Div(
                    f"Callback script {callback_filename} saved to 'visualizer/callbacks/' directory.",
                    style={"color": "green"},
                )
            )

        return html.Ul([html.Li(msg) for msg in status_messages])
