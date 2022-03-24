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
from plotly.subplots import make_subplots
import sys
sys.path.append("../")
from utils import get_active_cells

app = dash.Dash(__name__)

# Import and clean data
BASE_PATH = "D:/Vid_148/"
cells_file = "cells.pkl"

with open(BASE_PATH + cells_file, 'rb') as f:
    cells = pickle.load(f)

cells = get_active_cells(cells)
keys_as_list = list(cells.keys())
max_cell_number = max(keys_as_list)

# app
app.layout = html.Div(
    [
        html.H1("Traces per trial type (vid148)", style={'text-align':'center'}),
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

    fig = make_subplots(rows=6, cols=9,horizontal_spacing=0,vertical_spacing=0)

    this_cell_ID = keys_as_list[cell_slct-1]
    this_cell_trace = cells[this_cell_ID]['traces']
    colors = ['rgb(139,0,139)', 'rgb(0,206,209)', 'rgb(0,250,154)', 'rgb(255,69,0)','rgb(0,0,205)']

    coln_counter = 1
    for freq in this_cell_trace:

        row_counter = 1
        for intensity in reversed(this_cell_trace[freq]):

            rep_counter=0
            for repetition in this_cell_trace[freq][intensity]:

                trace = this_cell_trace[freq][intensity][repetition]
                trace = trace[5:]
                
                fig.append_trace(go.Scatter(
                    x=list(range(len(trace))),
                    y = trace,
                    line=dict(color=colors[rep_counter])
                    ),
                    row=row_counter,
                    col=coln_counter)

                rep_counter +=1

                # fig.update_layout(yaxis={'visible': False, 'showticklabels': False},row=row_counter,col=coln_counter)
                # fig.update_layout(xaxis={'visible': False, 'showticklabels': False},row=row_counter,col=coln_counter)
                fig.update_yaxes(range=[0,1100],showticklabels=False,row=row_counter,col=coln_counter,showgrid=False)
                fig.update_xaxes(showticklabels=False,row=row_counter,col=coln_counter,showgrid=False)

            row_counter += 1

        coln_counter += 1

    # fig.add_trace(go.Scatter(x=frames,y=this_cell_avg_trace, name="Average trace"))
    # fig.add_vline(x=5)
    # fig.update_layout(title="Averaged cell activity",xaxis_title="Frame", yaxis_title="dF/F")

    fig.update_layout(height=800,width=1600,showlegend=False) #,xaxis=dict(showgrid=False),yaxis=dict(showgrid=False))

    return container,fig

if __name__=="__main__":
    app.run_server(debug=True)