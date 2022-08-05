from xmlrpc.server import DocCGIXMLRPCRequestHandler
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pickle
import numpy as np
import pandas as pd

app = dash.Dash(__name__)

# Import and clean data
BASE_PATH = "D:/vid127_pseudorandom_stim/"
avg_traces_file = "active_cell_averages.pkl"

with open(BASE_PATH + avg_traces_file, 'rb') as f:
    traces = pickle.load(f)

keys_as_list = list(traces.keys())
max_cell_number = len(keys_as_list)

# app
app.layout = html.Div(
    [
        html.H1("Significantly active cells (vid127)", style={'text-align':'center'}),
        dcc.Input(id="cell_slct", type="number", 
                    min=1, max=max_cell_number, step=1, value=1,
                    placeholder="cell"),
        html.Div(id="output", children=[]),
        html.Br(),

        dcc.Graph(id='my_graph', figure={})
    ]
)

# Connect with Dash components
@app.callback(
    [Output(component_id="output",component_property="children"),
    Output(component_id="my_graph", component_property="figure")],
    [Input(component_id="cell_slct",component_property="value")],
)

def update_graph(cell_slct):
    container =  "The cell selected was: {}".format(keys_as_list[cell_slct-1])

    # get the average trace for this cell
    this_cell_ID = keys_as_list[cell_slct-1]
    this_cell_avg_trace = traces[this_cell_ID]
    frames = list(range(1,len(this_cell_avg_trace)+1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=frames,y=this_cell_avg_trace, name="Average trace"))
    fig.add_vline(x=5)
    fig.update_layout(title="Averaged cell activity",xaxis_title="Frame", yaxis_title="dF/F")

    return container,fig

if __name__=="__main__":
    app.run_server(debug=True)