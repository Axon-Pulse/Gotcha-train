import base64
import os
import sys

# from dotenv import load_dotenv, dotenv_values
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import torch
import yaml
from dash import Input, Output, callback, callback_context, dcc, html
from dash.dependencies import ALL, Input, Output, State
from dash.exceptions import PreventUpdate
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

"""
this callback contains the main callback functions for the data analysis page.
'display_raw_sample': used to display the raw and processed images of the selected sample.
'display_transforms_values': used to display the transforms values based on the selected transforms from the checklist.
'update_transforms': used to update the values of the transforms in the defaults.yaml file.
"""

import visualizer.main_callbacks
import visualizer.main_layout
from src.datamodule.isar_datamodule import TransformsWrapper
from visualizer.layouts.data_analysis import DataAnalysisLayout

"""
this callback contains the main callback functions for the data analysis page.
'update_transforms': used to update the values of the transforms in the defaults.yaml file, based on the user input and the selected transforms.
'display_transforms_values': used to display the transforms values based on the selected transforms from the checklist.
'display_raw_sample': used to display the raw and processed images of the selected sample.
 """


#  temporary function to extract the values of the transforms to dictionary format:
def extract_values(transform):
    transforms_inputs = {}
    for trans_name, trans_param in transform.items():
        if trans_name == "_target_":
            continue
        else:
            transforms_inputs[trans_name] = trans_param
    return transforms_inputs


def strip_values(value):
    if value.startswith("["):  # for list values in the input
        elements = value[1:-1].split(",")
        if all("." in x.strip() for x in elements):
            value = [float(x.strip()) for x in elements]
        else:
            value = [int(x.strip()) for x in elements]
    elif value.startswith("("):  # for tuple values in the input
        elements = value[1:-1].split(",")
        if all("." in x.strip() for x in elements):
            value = tuple(float(x.strip()) for x in elements)
        else:
            value = tuple(int(x.strip()) for x in elements)
    elif (
        value.replace(".", "", 1).isdigit() and value.count(".") < 2
    ):  # for float values in the input
        value = float(value)
    elif value.isdigit():  # for integer values in the input
        value = int(value)
    else:  # for text values in the input
        value = value
    return value


# Callback to apply the transforms on the data:
def register_callbacks(app, visualizer_instance):
    @app.callback(
        [Output("output", "children"), Output("transforms-store", "data")],
        Input("apply-transform-button", "n_clicks"),
        State("transforms-values", "children"),
        State("transforms-store", "data"),
    )
    def update_transforms(n_clicks, existing_inputs, trans_store):
        if n_clicks == 0:
            raise PreventUpdate
        else:
            trans_list = visualizer_instance.transforms
            trans_dict = {"order": []}
            for trans in existing_inputs:
                trans_name = trans["props"]["id"]
                if trans_name == "ToTensor":
                    trans_dict["order"].append(trans_name)
                    trans_dict[trans_name] = {"_target_": trans_list[trans_name]["_target_"]}
                    continue
                trans_params = [
                    trans["props"]["children"][param]["props"]["children"][0]["props"]["children"][
                        :-1
                    ]
                    for param in range(1, len(trans["props"]["children"]))
                ]
                trans_values = [
                    trans["props"]["children"][value]["props"]["children"][1]["props"]["value"]
                    for value in range(1, len(trans["props"]["children"]))
                ]
                for param, value in zip(trans_params, trans_values):
                    param_val = strip_values(value)
                    if trans_name not in trans_dict:
                        trans_dict[trans_name] = {"_target_": trans_list[trans_name]["_target_"]}
                    if trans_name not in trans_dict["order"]:
                        trans_dict["order"].append(trans_name)
                    trans_dict[trans_name][param] = param_val
            trans_store = trans_dict
            visualizer_instance.trans_dict = trans_dict
            div = (
                html.Div(html.H5("Transforms Applied!"), style={"color": "green"})
                if existing_inputs
                else html.Div(html.H5("No Transforms Selected!"), style={"color": "red"})
            )
        return div, trans_store

    # Callback to display the transforms values based on the selected transforms from the checklist:
    @app.callback(
        Output("transforms-values", "children"),
        Input("transforms-checklist", "value"),
        State("transforms-values", "children"),
    )
    def display_transforms_values(selected_transforms, existing_inputs):
        if not existing_inputs:
            existing_inputs = {}

        inputs = []
        inputs_dict = {input["props"]["id"]: input for input in existing_inputs}
        for index, trans_name in enumerate(selected_transforms):
            default_values = extract_values(visualizer_instance.transforms[trans_name])
            trans_id = f"{trans_name}"
            if trans_id in inputs_dict:
                inputs.append(inputs_dict[trans_id])
            else:
                transform_div = html.Div(
                    id=trans_id,
                    children=[
                        html.H5(f"{index+1}. {trans_name}"),
                        *[
                            html.Div(
                                [
                                    html.Label(f"{param_name}:"),
                                    dcc.Input(
                                        id=f"{trans_name}.{param_name}",
                                        value=str(value),
                                        type="text",
                                    ),
                                ]
                            )
                            for param_name, value in default_values.items()
                            if default_values
                        ],
                    ],
                    style={"marginBottom": "10px"},
                )
                inputs.append(transform_div)
                if isinstance(existing_inputs, list):
                    existing_inputs.append(transform_div)
                elif isinstance(existing_inputs, dict):
                    existing_inputs[trans_id] = transform_div
        return inputs

    # Callback to display the raw and processed images of the selected sample:
    @app.callback(Output("content-container", "children"), Input("visualization-type", "value"))
    def update_layout(visualization_type):
        if visualization_type == "img":
            visualizer_instance.vis_type = "img"
            return html.Div(
                id="image-div",
                children=[
                    dbc.Row(
                        [  # This row contains both graphs side by side
                            dbc.Col(
                                dcc.Graph(id="original-image-graph", style={"height": "50vh"}),
                                width=6,
                            ),
                            dbc.Col(
                                dcc.Graph(id="transformed-image-graph", style={"height": "50vh"}),
                                width=6,
                            ),
                        ]
                    ),  # Additional components specific to image processing
                ],
                style={"display": "none"},
            )
        elif visualization_type == "grph":
            visualizer_instance.vis_type = "grph"
            return html.Div(
                id="graph-div",
                children=[
                    dcc.Graph(id="signal-graph"),
                    dcc.Graph(id="rpr_gt-graph"),
                ],
                style={"display": "none"},
            )
        else:
            return "Select a valid visualization type"

    # Callback to display the raw and processed images of the selected sample:
    @app.callback(
        [
            Output("original-image-graph", "figure"),
            Output("transformed-image-graph", "figure"),
            Output("model-output", "figure"),
            Output("model-output", "style"),
            Output("transformed-image-graph", "style"),
            Output("image-div", "style"),
        ],
        Input("table-data", "selected_rows"),
        Input("apply-transform-button", "n_clicks"),
        State("transforms-store", "data"),
        prevent_initial_call=True,
    )
    def display_raw_sample(selected_row, n_clicks, trans_dict):
        if visualizer_instance.vis_type == "img" and selected_row:
            dataset = visualizer_instance.dataset
            df = dataset.dataframe
            label_gt = df["label"].unique()
            trans_dict = visualizer_instance.trans_dict
            for row_idx in selected_row:
                index_id = df.iloc[row_idx]["id"]
                dataset.transformation = None
                org_fig = dataset.__plotsample__(row_idx)
                org_fig.update_layout(
                    title=f"{index_id} Original Image: ", margin=dict(l=10, r=10, t=30, b=10)
                )
                if n_clicks and trans_dict["order"]:
                    build_transform = TransformsWrapper(trans_dict)
                    trns = dataset.update_transformation(build_transform)
                    trans_fig = dataset.__plotsample__(row_idx)
                    trans_fig.update_layout(
                        title=f"{index_id} Transformed Image: ",
                        margin=dict(l=10, r=10, t=30, b=10),
                    )
                    pred_fig = dataset.__getmodel__(
                        row_idx, visualizer_instance.model
                    )  # get_predictions(dataset, label_gt, row_idx)
                    return (
                        org_fig,
                        trans_fig,
                        pred_fig,
                        {"height": "60vh", "width": "60vh", "display": "block"},
                        {"height": "50vh", "display": "block"},
                        {"display": "block"},
                    )
                else:
                    pred_fig = org_fig
                    trans_fig = org_fig
                    return (
                        org_fig,
                        trans_fig,
                        pred_fig,
                        {"display": "none"},
                        {"height": "60vh", "width": "60vh", "display": "none"},
                        {"display": "block"},
                    )
        else:
            raise PreventUpdate

    @app.callback(
        Output("signal-graph", "figure"),
        Output("rpr_gt-graph", "figure"),
        Output("model-output", "children"),
        Output("model-output", "style"),
        Output("graph-div", "style"),
        [Input("table-data", "selected_rows"), Input("apply-transform-button", "n_clicks")],
        prevent_initial_call=True,
    )
    def plot_signals(selected_row, n_clicks):
        if visualizer_instance.vis_type == "grph" and selected_row:
            dataset = visualizer_instance.dataset
            trans_dict = visualizer_instance.trans_dict
            signal_traces = []
            rpr_gt_traces = []
            for row_idx in selected_row:
                signal_trace, rpr_gt_trace = dataset.__plotsample__(row_idx)
                signal_traces.append(signal_trace)
                rpr_gt_traces.append(rpr_gt_trace)
            signal_fig = go.Figure(data=signal_traces)
            rpr_gt_fig = go.Figure(data=rpr_gt_traces)
            signal_fig.update_layout(
                title="Absolute Values of Signal",
                xaxis={"title": "Index"},
                yaxis={"title": "Absolute Value"},
                hovermode="closest",
            )
            rpr_gt_fig.update_layout(
                title="Absolute Values of Rpr_gt",
                xaxis={"title": "Index"},
                yaxis={"title": "Absolute Value"},
                hovermode="closest",
            )
            if n_clicks and selected_row:
                build_transform = TransformsWrapper(trans_dict)
                trns = dataset.update_transformation(build_transform)
                signal_model_output, rpr_model_output = dataset.__getmodel__(
                    row_idx, visualizer_instance.model
                )
                children = html.Div([signal_model_output, rpr_model_output])
            return (
                signal_fig,
                rpr_gt_fig,
                children,
                {"height": "60vh", "width": "60vh"},
                {"display": "block"},
            )
        else:
            raise PreventUpdate
