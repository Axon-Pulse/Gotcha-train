from dash import dcc, html
import dash_bootstrap_components as dbc
import yaml

# from visualizer.main_layout import Visualization
import visualizer.main_callbacks


""""
this is the layout function of the data analysis page. it contains the original and transformed images of the selected row in the 
data table.
"""



# visualizer_instance = Visualization()

class DataAnalysisLayout:
    def __init__(self):
        # default_transforms = visualizer_instance.default_transforms

        # self.transforms_list = visualizer_instance.transforms
        return
    

    def data_analysis_layout(self, transforms, default_transforms):
        self.transforms = transforms
        self.default_transforms = default_transforms
        self.predict_trans = default_transforms['valid_test_predict']
        import visualizer.callbacks.data_analysis_callback as callback
        layout = html.Div([
            html.Div([
                html.H3("Graphs Visualization",style={'textAlign': "center"}),  
                html.Div(id='content-container'),
                dcc.Graph(id='model-output', style={'display': 'none'}),
                ], className="content"
                ),
            html.Div([
                html.H2("Data Analysis"),
                html.Hr(),
                html.H4("Graphs Visualization"),
                dcc.Markdown('''
                ###### Here you can visualize the original and transformed images of the selected sample, follow the instructions:
                             
                1. Select the type of the visualization plots in your data.
                             
                2. Select the sample from the table.
                             '''),
                dcc.RadioItems(id='visualization-type',
                               options=[
                                    {'label': 'Images', 'value': 'img'},
                                    {'label': 'Graphs', 'value': 'grph'}
                                ],
                                labelStyle={'display': 'inline-block', 'marginRight': '20px'}  # Display options in a horizontal line
                            ),
                html.Hr(),
                html.H4("Transforms Selection") , 
                dcc.Markdown('''
                ###### Here you can select and Edit the transforms you want to implement on the images/graphs, follow the instructions:
                            
                1. Select the transforms you want, edit the parameter's values.
                             
                **Notice:** Select the Transforms in the order you want them to be implemented.
                             
                2. Select the sample from the table.
                3. Click on 'Apply Transforms' to activate the transforms on the selected sample. 
                '''),
                html.Hr(),
                dbc.Container([            
                dbc.Row([dbc.Col([html.H4('Transform Names'),
                                dbc.Checklist(id= 'transforms-checklist',
                                        options=[{'label': key, 'value': key} for key in self.transforms.keys()],
                                        value=[selected for selected in self.predict_trans['order']])]),
                        dbc.Col([html.H4('Transforms Values'),
                        html.Div(id='transforms-values')]),
                ]),
                html.Button('Apply Transforms', id='apply-transform-button', n_clicks=0),
                
            dcc.Store(id='transforms-store'),
            html.Div(id='output')
            ],id='transforms-list_display'),
            ], className="sidebar")
            ])
            # # Dropdown to select the model checkpoint
            # dcc.Dropdown(id='checkpoint-dropdown', placeholder="Select checkpoint"),
            
            # # Display the model output (e.g., bar plot of probabilities)
            # dcc.Graph(id='model-output'),
            
            # # Display Grad-CAM results
            # html.Div(id='gradcam-output')
        return layout   
     
