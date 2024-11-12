from dash import html, dcc, callback_context , callback
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from dash.exceptions import PreventUpdate
import pandas as pd
import ast
import re
import json
import base64

import visualizer.main_callbacks
import visualizer.layouts.data_visualization
import visualizer.callbacks.data_visualization_callback
'''
This file is for the data graphs- import, export, update, creation, and design.
'add_graph', 'display_graph': used to add the graph creation blocks based on the number of graph selected by the user.
'update_data_graph': used to display the data-graph.
'update_histogram': used to display the histogram.
'update_bar_plot', 'update_facet': used to display the bar plot.
'update_facet': used to update the facet input fields.

'create_import_layout': used to create the layout when importing data.
'download_graph_config': used to download the graphs.
'show_import': used to update the suitable layout from the imported data.

In addition we have here also a helper functions for the processes:
'create_data_graph_layout', 'create_histogram_layout', 'create_bar_plot_layout': used to create the layout of the graph, histogram,
 and bar plot.
'generate_barplot': used to create the bar plot.
'importing_process': used to import the data from the json file.
'import_layout': used to create the layout when importing data.
** Its crucial that there will be stores for the data-graph, histogram, and bar plot for each graph. even if it will be empty in this
 index, in order to manage the places of the graphs when importing them.

'''
#  Callbacks for the data visualization page if selecting the number of graphs:
def register_callbacks(app, visualizer_instance):

    @app.callback(
        Output("graph-container", "children",allow_duplicate = True),
        
        Input("num-graphs-dropdown", "value"),   
        Input('import-btn', 'n_clicks'),

        State('graph-container', 'children'),
        prevent_initial_call=True
    )
    def add_graph(num_graphs, n_clicks, existing_graphs):
        '''
        This function is used to add a new graph, based on the number of graphs selected by the user.
        it also includes the data-graph, histogram, and bar plot menus layout, and the stores.
        '''
        # Clear the graph container before adding new graphs
        ctx = callback_context
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        if existing_graphs and triggered == 'import-btn':
            visualizer_instance.imports = True
            return []  # Clear the container before importing
        elif triggered == 'num-graphs-dropdown':
            visualizer_instance.imports = False  # Reset import flag
            if num_graphs is not None:
                children = []
                for i in range(num_graphs):
                    new_child = html.Div([
                        html.H4(f"Graph {i+1}:"),
                        html.Label("Select the graph type:"),
                        dcc.Dropdown(
                            id={'type': 'dynamic-dropdown', 'index': i},
                            options=[{'label': 'Data Graph', 'value': 'data-graph'},
                                    {'label': 'Histogram', 'value': 'histogram'},
                                    {'label': 'Bar Plot', 'value': 'bar-plot'}],
                            style={'width': '70%'}
                        ),
                        html.Div(id= {'type': 'dynamic-layout','index': i}),
                        html.Div(id= {'type': 'import-layout','index': i},style = {'display': 'none'}),
                        dcc.Graph(id={'type': 'import-graph', 'index': i}, style={"height": "60vh", 'width':'60vh', 'borderRight': '3px solid RoyalBlue','display':'none'}),
                        dcc.Graph(id={'type': 'dynamic-graph', 'index': i}, style={"height": "60vh", 'width':'60vh', 'borderRight': '3px solid RoyalBlue','display':'none'}),                
                        dcc.Store(id={'type': 'data_graph-store','index': i}, data={}),
                        dcc.Store(id={'type': 'histogram-store','index': i}, data={}),
                        dcc.Store(id={'type': 'bar_plot-store','index': i}, data={}),
                    ]
                    , id='dynamic-div', style={'width': '30%', 'minWidth': '300px', 'padding': '10px'})
                    children.append(new_child)
                return children
            else:
                return []

    # Callback to download the graphs:
    @app.callback(
        Output("download-graphs", "data"),
        
        Input("export-button", "n_clicks"),
        
        State({'type':'data_graph-store', 'index': ALL}, 'data'),
        State({'type':'histogram-store', 'index': ALL}, 'data'),
        State({'type':'bar_plot-store', 'index': ALL}, 'data'),
        prevent_initial_call=True
    )
    def download_graph_config(n_clicks, data_graph_store, histogram_store, bar_plot_store):
        '''
        The function download_graph_config generates a JSON file containing data from three sources 
        (data_graph_store, histogram_store, bar_plot_store). It names the file using the current date
        '''
        from datetime import date
        today = date.today()
        day = today.strftime("%d%m%Y")
        if n_clicks:
            data_to_save = {'data graph': data_graph_store,'histogram': histogram_store, 'bar plot': bar_plot_store}
        return dict(content=json.dumps(data_to_save), filename=f"all_graphs_{day}.json")

    # Callback to create the import layout:
    @app.callback(
        Output("graph-container", "children", allow_duplicate = True),
        Output("full data", "data"),

        Input('import-upload', 'contents'),

        State('full data', 'data'),
        State( 'graph-container', 'children'),
        prevent_initial_call=True
    )
    def create_import_layout(imported_data, full_data_store, existing_graphs):
        '''
        This function is used to create the layout when importing data.
        First, it imports the data from the json file, extracts the data table, 
        and creates the layout for each graph.
        'import_layout': the layout to create when importing the data. 
        It includes the data-graph, histogram, and bar plot menus layout, and stores.
        '''
        if visualizer_instance.flag:
            raise PreventUpdate    
        visualizer_instance.flag = True # To prevent repeating the process of importing the data

        ctx = callback_context
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if triggered == 'import-upload':
            visualizer_instance.imports = True
            children =[]
            num_graphs, full_data = importing_process(imported_data)
            filtered_data = next((item['data'] for item in full_data['data_graph'] if 'data' in item), None) # Get the data table from the imported data
            data_df = pd.DataFrame(filtered_data)
            full_data_store = full_data

            if imported_data is not None:
                for i in range(num_graphs): # Create the layout for each graph based on the number of graphs in the imported data
                    new_child = import_layout(i,data_df)
                    graphs = html.Div([
                        dcc.Graph(id={'type': 'import-graph', 'index': i}, style={"height": "60vh", 'width':'60vh','display':'none'}),
                        dcc.Graph(id={'type': 'dynamic-graph', 'index': i}, style={"height": "60vh", 'width':'60vh','display':'none'})
                        ])
                    elements = [new_child,graphs]
                    children.extend(elements)
                return children, full_data_store
            else:
                return [], {}

    # Callback to import the data and plot the suitable layout:
    @app.callback(  
        Output({'type':'data_graph-store', 'index': MATCH}, 'data',allow_duplicate = True),
        Output({'type':'histogram-store', 'index': MATCH}, 'data',allow_duplicate = True),
        Output({'type':'bar_plot-store', 'index': MATCH}, 'data',allow_duplicate = True),    

        Input("show-graph", "n_clicks"),

        State('full data', 'data'),
        prevent_initial_call = True,
        )
    def show_import(n_clicks, full_data):
        '''
        This function is used to update the suitable graph's layout from the imported data.
        '''
        if not n_clicks:
            raise PreventUpdate
        data_graph = full_data.get('data_graph',[])
        histogram = full_data.get('histogram',[])
        bar_plot = full_data.get('bar plot',[])
        max_len = max(len(data_graph), len(histogram), len(bar_plot))
        for i in range(max_len):
            if  i<len(data_graph) and data_graph[i]:
                data_graph[i]['import_flag'] = 1
            if  i<len(histogram) and histogram[i]:
                histogram[i]['import_flag'] = 1
            if i<len(bar_plot) and bar_plot[i]:
                bar_plot[i]['import_flag'] = 1
        return data_graph, histogram, bar_plot

    # Callback to present the desired graph style:
    @app.callback(
        Output({'type': 'dynamic-layout', 'index': MATCH}, 'children'),
        Output({'type': 'dynamic-layout', 'index': MATCH}, 'style'),
        
        Input({'type': 'dynamic-dropdown', 'index': MATCH}, 'value'),
        
        State({'type': 'dynamic-layout', 'index': MATCH}, 'id'),
        State('filtered-data-table', 'data'),  
    )
    def display_graph(selected_graph, layout_id,filtered_data):
        '''
        This function is used to display the desired graph style, based on the selected graph by the user.
        '''

        layout = None
        if visualizer_instance.imports:
            raise PreventUpdate
        if not selected_graph:
            raise PreventUpdate
        data_df = pd.DataFrame(filtered_data)
        if selected_graph == 'data-graph':
            layout = create_data_graph_layout(layout_id, data_df)
        elif selected_graph == 'histogram':
            layout = create_histogram_layout(layout_id, data_df)
        elif selected_graph == 'bar-plot':
            layout = create_bar_plot_layout(layout_id, data_df)
        return layout, {'display': 'block'}

    # Callback to display the data-graph:
    @app.callback(
        Output({'type': 'dynamic-graph', 'index': MATCH}, 'figure',allow_duplicate = True),
        Output({'type': 'dynamic-graph', 'index': MATCH}, 'style',allow_duplicate = True),
        Output({'type': 'import-graph', 'index': MATCH}, 'figure',allow_duplicate = True),
        Output({'type': 'import-graph', 'index': MATCH}, 'style',allow_duplicate = True),
        Output({'type':'data_graph-store', 'index': MATCH}, 'data'),
        
        Input({'type': 'data_graph-btn', 'index': ALL}, 'n_clicks'),
        Input({'type':'data_graph-store', 'index': MATCH}, 'data'),
        
        State({'type':'data_graph-store', 'index': MATCH},'id'),
        State({'type': 'y_axis-selector','index': MATCH}, 'value'),   
        State({'type':'x_axis-selector','index': MATCH}, 'value'), 
        State('filtered-data-table', 'data'),
        prevent_initial_call=True
    )
    def update_data_graph(generate_btn, graph_store,ids, y_axis, x_axis, filtered_data):
        '''
        This function is used to display the data graph.
        If the graph is newly created, it will display the graph based on the selected x and y axes.
        But if the graph is imported, it will display the graph based on the imported data.
        'import_flag': a flag to indicate if the graph is imported or newly created.

        '''
        if not generate_btn and not graph_store:
            raise PreventUpdate
        index= ids['index']
        if isinstance(graph_store, dict):  # Handling for newly created graphs
            # Process the dictionary directly
            import_flag = graph_store.get('import_flag', None)
        elif isinstance(graph_store, list):  # Handling for imported graphs
            import_flag = next((item['import_flag'] for item in graph_store if 'import_flag' in item), None)
        
        if import_flag == 1:
            for i, dg in enumerate(graph_store):
                if i==index and dg: # In order to manage the graph's places its crucial to verify that
                                    #  the index of the graph's place is equal to the index of the loop
                                    #  and also it have data in the store  
                    try:
                        filtered_data = pd.DataFrame(dg['data'])                
                        x_axis = dg['axes'][0]
                        y_axis = dg['axes'][1]
                        dg['import_flag'] = 0
                        graph = px.scatter(filtered_data, x=filtered_data[x_axis], y=filtered_data[y_axis],title=f'{y_axis} vs {x_axis}')
                        graph.update_layout(xaxis_type='category')
                        max_x = filtered_data[x_axis].max() if pd.api.types.is_numeric_dtype(filtered_data[x_axis]) else len(filtered_data[x_axis].unique()) - 0.5
                        graph.add_shape(
                            type="line",
                            x0=max_x,
                            y0=0,
                            x1=max_x,
                            y1=graph.layout.yaxis.range[1] if graph.layout.yaxis.range else filtered_data[y_axis].max(),
                            line=dict(color="RoyalBlue", width=3)
                        )                
                        visualizer_instance.imports = False # To enable the user from adding new graphs
                        visualizer_instance.flag = False # To enable repeating the process of importing the data
                        return graph, {'display': 'none'}, graph, {'display': 'block', 'borderRight': '3px solid RoyalBlue','marginRight': '5px', 'height': '60vh', 'width': '60vh'}, graph_store
                    except Exception as e:
                        print("Error rendering imported data graph:", e)
                        raise PreventUpdate    
                else: # In case of empty graph in the index of the loop it will iterate to the next index
                    print(f'index {i} is an empty data graph')
        else:
            try:
                filtered_data = pd.DataFrame(filtered_data)
                graph = px.scatter(filtered_data, x=filtered_data[x_axis], y=filtered_data[y_axis],title=f'{y_axis} vs {x_axis}')
                graph.update_layout(xaxis_type='category')
                updated_store = {'axes': [x_axis, y_axis], 'data': filtered_data.to_dict('records'), 'import_flag': 0}
                return graph, {'display': 'block', 'borderRight': '3px solid RoyalBlue','marginRight': '5px', 'height': '60vh', 'width': '60vh'}, graph, {'display': 'none'}, updated_store
            except Exception as e:
                print("Error rendering new data graph:", e)
                raise PreventUpdate

    # Callback to display the histogram:
    @app.callback(
        Output({'type': 'dynamic-graph', 'index': MATCH}, 'figure',allow_duplicate = True),
        Output({'type': 'dynamic-graph', 'index': MATCH}, 'style',allow_duplicate = True),
        Output({'type': 'import-graph', 'index': MATCH}, 'figure',allow_duplicate = True),
        Output({'type': 'import-graph', 'index': MATCH}, 'style',allow_duplicate = True),
        Output({'type': 'histogram-store', 'index': MATCH}, 'data'),
            
        Input({'type': 'hist-btn', 'index': MATCH}, 'n_clicks'),
        Input({'type':'histogram-store', 'index': MATCH}, 'data'),

        State({'type': 'histogram-store', 'index': MATCH}, 'id'),
        State({'type': 'histogram-selector','index': MATCH}, 'value'),
        State('filtered-data-table', 'data'),
        prevent_initial_call=True
    )
    def update_histogram(generate_btn, histogram_store, ids, selected_column, filtered_data):
        '''
        This function is used to display the histogram of the selected column.
        'selected_column': the column to display the histogram.
        If the graph is newly created, it will display the graph based on the selected column.
        But if the graph is imported, it will display the graph based on the imported data.
        'import_flag': a flag to indicate if the graph is imported or newly created.

        '''
        if not generate_btn and not histogram_store:
            raise PreventUpdate
        index = ids['index']
        if isinstance(histogram_store, dict):  # Handling for newly created graphs
            # Process the dictionary directly
            import_flag = histogram_store.get('import_flag', None)
        elif isinstance(histogram_store, list):  # Handling for imported graphs
            import_flag = next((item['import_flag'] for item in histogram_store if 'import_flag' in item), None)
        
        if import_flag == 1:
            for i, hist in enumerate(histogram_store):
                if i==index and hist: # In order to manage the graphs places its crucial to verify that the index 
                                    # of the graph's place is equal to the index of the loop and also it have data
                                    #  in the store
                    try:
                        filtered_data = pd.DataFrame(hist['data'])
                        selected_column = hist['axes']
                        hist['import_flag'] = 0
                        graph = px.histogram(filtered_data[selected_column], x=selected_column)
                        graph.update_layout(xaxis_type='category')
                        max_x = filtered_data[selected_column].max() if pd.api.types.is_numeric_dtype(filtered_data[selected_column]) else len(filtered_data[selected_column].unique()) - 0.5
                        graph.add_shape(
                            type="line",
                            x0=max_x,
                            y0=0,
                            x1=max_x,
                            y1=graph.layout.yaxis.range[1] if graph.layout.yaxis.range else filtered_data[selected_column].value_counts().max(),
                            line=dict(color="RoyalBlue", width=3)
                        )
                        visualizer_instance.imports = False # To enable the user from adding new graphs
                        visualizer_instance.flag = False # To enable repeating the process of importing the data
                        return graph, {'display': 'none'}, graph, {'display': 'block', 'borderRight': '3px solid RoyalBlue','marginRight': '5px', 'height': '60vh', 'width': '60vh'}, histogram_store
                    except Exception as e:
                        print(f"Error rendering imported histogram for index {i}:", e)
                        raise PreventUpdate       
                else:
                    print(f'index {i} is an empty histogram')
        else: 
            try:
                filtered_data = pd.DataFrame(filtered_data)
                graph = px.histogram(filtered_data, x=selected_column, title = f"Histogram of: {selected_column}")
                graph.update_layout(xaxis_type='category')
                updated_store = {'axes': selected_column, 'data': filtered_data.to_dict('records'), 'import_flag': 0}
                return graph, {'display': 'block', 'borderRight': '3px solid RoyalBlue','marginRight': '5px', 'height': '60vh', 'width': '60vh'}, graph, {'display': 'none'}, updated_store        
            except Exception as e:
                print("Error rendering new histogram:", e)
                raise PreventUpdate
            

    # Callback to display the facet input fields:
    @app.callback(
        Output({'type': 'facet-input','index': MATCH}, 'children'),
        
        Input({'type': 'facet-checklist','index':MATCH}, 'value'),
        
        State({'type': 'dynamic-layout', 'index': MATCH}, 'id'),
        State('filtered-data-table', 'data')
    )
    def update_facet(operation,layout_id, data):
        '''
        This function is used to display the facet input fields.    
        '''
        if operation is None:
            raise PreventUpdate
        data_df=pd.DataFrame(data)
        inputs = []
        for opr in operation:
            dcc_obj = html.Div(id = {'type': f'{opr}-div', 'index': layout_id['index']}, children =
                                [html.Label(f"Enter {opr} column:"),
                                dcc.Dropdown(id={'type':f'{opr}', 'index': layout_id['index']},
                                            options=[{'label': col, 'value': col} for col in data_df.columns], 
                                            style={'width': '100%'})])
            inputs.append(dcc_obj)
        return inputs

    # Callback to update the bar plot:
    @app.callback(
        Output({'type': 'dynamic-graph', 'index': MATCH}, 'figure',allow_duplicate = True),
        Output({'type': 'dynamic-graph', 'index': MATCH}, 'style',allow_duplicate = True),
        Output({'type': 'import-graph', 'index': MATCH}, 'figure',allow_duplicate = True),
        Output({'type': 'import-graph', 'index': MATCH}, 'style',allow_duplicate = True),
        Output({'type':'bar_plot-store', 'index': MATCH}, 'data'),
            
        Input({'type': 'barplot-btn', 'index': MATCH}, 'n_clicks'),
        Input({'type':'facet-input','index': MATCH}, 'children'),
        Input({'type':'bar_plot-store', 'index': MATCH}, 'data'), 
        
        State({'type':'bar_plot-store', 'index': MATCH}, 'id'), 
        State({'type': 'groupby-selector', 'index': MATCH}, 'value'),
        State({'type': 'count-by-selector', 'index': MATCH}, 'value'),
        State({'type': 'color-by-selector', 'index': MATCH}, 'value'),
        State({'type': 'facet-checklist', 'index': MATCH}, 'value'),
        State('filtered-data-table', 'data'),
        prevent_initial_call=True
    )
    def update_bar_plot(generate_btn, facet_inputs, bar_plot_store,ids, group_by, count_by,color_by, facet_type, filtered_data):
        '''
        This function is used to update the bar plot.
        If the graph is newly created, it will display the graph based on the selected x and y axes.
        But if the graph is imported, it will display the graph based on the imported data.
        'import_flag': a flag to indicate if the graph is imported or newly created.

        '''    
        if not generate_btn and not bar_plot_store:
            raise PreventUpdate
        index = ids['index']
        if isinstance(bar_plot_store, dict):  # Handling for newly created graphs
            # Process the dictionary directly
            import_flag = bar_plot_store.get('import_flag', None)
        elif isinstance(bar_plot_store, list):  # Handling for imported graphs
            import_flag = next((item['import_flag'] for item in bar_plot_store if 'import_flag' in item), None)
        if import_flag == 1:
            for i, bp in enumerate(bar_plot_store):
                if i==index and bp: # In order to manage the graphs places its crucial to verify that the index of
                                    #  the graph's place is equal to the index of the loop and also it have data in
                                    #  the store
                    try:
                        data_df = pd.DataFrame(bp['data'])
                        group_by = bp['axes'][0]
                        count_by = bp['axes'][1]
                        facet_row = bp['axes'][2]
                        facet_col = bp['axes'][3]
                        color_by = bp['axes'][4]
                        data_df['group_combination'] = data_df[group_by].astype(str).agg(', '.join, axis=1)
                        group_columns = ['group_combination'] + group_by + [color_by] if color_by else ['group_combination'] + group_by
                        group_columns = group_columns + [count_by] if count_by else group_columns
                
                # Group the data by the selected columns and count occurrences
                        bar_table = data_df.groupby(group_columns).size().reset_index(name='count')
                        graph = generate_barplot(bar_table, count_by, facet_row, facet_col, group_by, color_by) # Function to create the bar plot with the selected inputs (if there are any)
                        bp['import_flag'] = 0
                        visualizer_instance.imports = False # To enable the user from adding new graphs
                        visualizer_instance.flag = False # To enable repeating the process of importing the data
                        return graph, {'display': 'none'} , graph, {'display': 'block', 'borderRight': '3px solid RoyalBlue','marginRight': '5px', 'height': '60vh', 'width': '60vh'}, bar_plot_store
                    except Exception as e:
                        print(f"Error rendering imported bar plot for index {i}:", e)
                        raise PreventUpdate     
                else:
                    print(f'index {i} is an empty bar plot')
        else: 
            try:
                data_df = pd.DataFrame(filtered_data)
                facet_col = None
                facet_row = None
                if facet_type: # Check if facet is selected and get the inputs to update the bar plot accordingly
                    inputs = [facet_inputs[i]['props']['children'][1]['props']['value'] for i in range(len(facet_type))]    
                    if 'facet_row' in facet_type:
                        facet_row = inputs[facet_type.index('facet_row')]
                        group_by.append(facet_row)
                    if 'facet_col' in facet_type:
                        facet_col = inputs[facet_type.index('facet_col')]
                        group_by.append(facet_col)
                
                data_df['group_combination'] = data_df[group_by].astype(str).agg(', '.join, axis=1)
                group_columns = ['group_combination'] + group_by + [color_by] if color_by else ['group_combination'] + group_by
                group_columns = group_columns + [count_by] if count_by else group_columns
                
                # Group the data by the selected columns and count occurrences
                bar_table = data_df.groupby(group_columns).size().reset_index(name='count')

                graph = generate_barplot(bar_table, count_by, facet_row, facet_col, group_by, color_by) # Function to create the bar plot with the selected inputs (if there are any)
                updated_store ={'axes': [group_by, count_by, facet_row,facet_col, color_by], 'data': data_df.to_dict('records'), 'import_flag': 0}
                return  graph, {'display': 'block', 'borderRight': '3px solid RoyalBlue','marginRight': '5px', 'height': '60vh', 'width': '60vh'}, graph, {'display': 'none'}, updated_store 
            except Exception as e:
                print("Error rendering new bar plot:", e)
                raise PreventUpdate

    ######################################################################################################################################

    # Helper functions:
    def create_histogram_layout(layout_id, df):
        '''
        This function is used to create the layout of the histogram.
        '''
        return html.Div([
            html.H2('Histogram'),
            html.Label("Select Column:"),
            dcc.Dropdown(id={'type': 'histogram-selector', 'index': layout_id['index']}, 
                        options=[{'label': col, 'value': col} for col in df.columns],
                        value = [], 
                        style={'width': '60%'}),
            html.Button('Generate Graph', id = {'type': 'hist-btn', 'index': layout_id['index']},
                        n_clicks = 0, style = {'margin_top':'20px'}),
        ])

    def create_data_graph_layout(layout_id, df):
        '''
        This function is used to create the layout of the data graph.
        '''
        layout = html.Div([
        html.H3('Data Graph'),
        html.Div([
            html.Div([
                html.Label("Select X-Axis:"),# style={'marginRight': '10px'}),
                dcc.Dropdown(id={'type': 'x_axis-selector', 'index': layout_id['index']}, 
                            options=[{'label': col, 'value': col} for col in df.columns],
                            value='id',
                            style={'width': '200px'}),  # Adjust width as needed
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '10px'}),
            html.Div([
                html.Label("Select Y-Axis:"),# style={'marginRight': '10px'}),
                dcc.Dropdown(id={'type': 'y_axis-selector', 'index': layout_id['index']}, 
                            options=[{'label': col, 'value': col} for col in df.columns],
                            value=[], 
                            style={'width': '200px'}),  # Adjust width as needed
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
        ], style={'textAlign': 'center'}),
        html.Button('Generate Graph', id={'type': 'data_graph-btn', 'index': layout_id['index']},
                    n_clicks=0, style={'marginTop': '20px', 'display': 'block', 'marginLeft': 'auto', 'marginRight': 'auto'}),
    ])#, style={'textAlign': 'center'})    
        return layout

    def create_bar_plot_layout(layout_id, df):
        '''
        This function is used to create the layout of the bar plot.
        '''
        return html.Div([
        html.H3('Bar Plot'),
        dbc.Row([dbc.Col(html.Div([
            dbc.Checklist(id={'type': 'facet-checklist', 'index': layout_id['index']}, 
                        options=[{'label': 'Facet Row', 'value': 'facet_row'},
                                {'label': 'Facet Column', 'value': 'facet_col'}], 
                        inline=True,  # This makes the checklist items display inline
                        style={'marginRight': '10px'})])),
                dbc.Col(html.Div(id={'type': 'facet-input', 'index': layout_id['index']}, style={'width': '100%'}))]),
        dbc.Row([
            dbc.Col(html.Label("Group by:", style={'marginRight': '10px'})),
            dbc.Col(html.Label("Count by (Optional):", style={'marginRight': '10px'})),
            dbc.Col(html.Label("Color by (Optional):", style={'marginRight': '10px'})),
        ]), #, style={'display': 'flex', 'marginBottom': '10px'}
        html.Div([
            dcc.Dropdown(id={'type': 'groupby-selector', 'index': layout_id['index']},
                        options=[{'label': col, 'value': col} for col in df.columns],
                        multi=True,
                        value=[], 
                        style={'width': '100%', 'marginRight': '3px', 'display': 'inline-block'}),
            dcc.Dropdown(id={'type': 'count-by-selector', 'index': layout_id['index']},
                        options=[{'label': col, 'value': col} for col in df.columns],
                        style={'width': '100%', 'marginRight': '3px', 'display': 'inline-block'}),
            dcc.Dropdown(id={'type': 'color-by-selector', 'index': layout_id['index']},
                        options=[{'label': col, 'value': col} for col in df.columns],
                        style={'width': '100%', 'marginRight': '3px', 'display': 'inline-block'}),
                ], style={'display': 'flex', 'alignItems': 'center'}),
        html.Button("Generate Graph", id={'type': 'barplot-btn', 'index': layout_id['index']}, n_clicks=0, style={'marginTop': '20px', 'display': 'block', 'marginLeft': 'auto', 'marginRight': 'auto'}),
    ])#, style={'textAlign': 'center'}) , 'justifyContent': 'center', , 'alignItems': 'center', 'justifyContent': 'center'

    def generate_barplot(bar_table, count_by, facet_row, facet_col, group_by, color_by):
        """
        Create the bar plot figure based on the selected inputs (if there are any).
        'bar_table': the data table to create the bar plot from.
        'count_by': the column to count by in the Bar plot.
        'facet_row': the column to facet row the Bar plot.
        'facet_col': the column to facet col the Bar plot.
        'group_by': the columns to group the Bar plot.
        """
        if count_by:
            y= count_by
        else:
            y='count'
        fig = px.bar(
            bar_table,
            x='group_combination',
            y=y,
            color=color_by if color_by else y,#group_by[1] if len(group_by) > 1 else group_by,
            text=color_by if color_by else y,#group_by[1] if len(group_by) > 1 else group_by,
            facet_row=facet_row,
            facet_col=facet_col,
            title=f"Bar plot of values grouped by: {', '.join(group_by)}"
        )

        return fig

    def import_layout(i,data_df):
        '''
        This function is used to create the layout when importing data.
        it includes the data-graph, histogram, and bar plot menus layout, and the stores.
        '''
        layout_id = {'index': i}
        return html.Div(id = 'import-div', children = [
                html.Div(id= {'type': 'dynamic-layout','index': i}),
                create_data_graph_layout(layout_id,data_df),
                create_histogram_layout(layout_id,data_df),
                create_bar_plot_layout(layout_id,data_df),
                dcc.Store(id={'type': 'data_graph-store','index': i}, data={}),
                dcc.Store(id={'type': 'histogram-store','index': i}, data={}),
                dcc.Store(id={'type': 'bar_plot-store','index': i}, data={}),
        ],style = {'display': 'none'})

    def importing_process(content):
        '''
        This function is used to import the data from the json file.
        '''
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        full_data = json.loads(decoded)
        
        data_graph = full_data.get('data graph',[])
        histogram = full_data.get('histogram',[])
        bar_plot = full_data.get('bar plot',[])
        max_len = max(len(data_graph), len(histogram), len(bar_plot))
        data_graph.extend([{}] * (max_len - len(data_graph)))
        histogram.extend([{}] * (max_len - len(histogram)))
        bar_plot.extend([{}] * (max_len - len(bar_plot)))    
        full_data = {'data_graph': data_graph, 'histogram': histogram, 'bar plot': bar_plot}
        return max_len, full_data
