#!/usr/bin/env python3

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import os
import numpy as np

import traceback
from pdroot.parse import to_ak_expr
import awkward1
from pdroot.readwrite import awkward1_arrays_to_dataframe

# BOOTSTRAP FLATLY COSMO LUMEN
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

a = awkward1.Array(
    dict(
        Jet_pt=[[42.0, 15.0, 10.5], [], [11.5], [50.0, 5.0]],
        Jet_eta=[[-2.2, 0.4, 0.5], [], [1.5], [-0.1, -3.0]],
        MET_pt=[46.5, 30.0, 82.0, 8.9],
    )
)
df = awkward1_arrays_to_dataframe(a)

app.layout = html.Div([
    # dcc.Markdown(df.to_markdown(tablefmt="grid").replace("[",r"\[").replace("]",r"\]")),
    html.I(" Input dataframe: "), 
    html.Pre(df.to_markdown(tablefmt="grid")),
    html.Hr(),
    html.Div([
        html.I(" Expression: "), 
        dcc.Input(
            id="expression",
            type="text",
            debounce=False,
            style = {"min-width": "40%"},
            value = "length(Jet_pt[Jet_pt > 40 and abs(Jet_eta) < 2.4])",
            autoFocus=True,
            autoComplete="false",
            persistence_type="session",
            ),
        ], style={'columnCount': 1}),
    html.Hr(),
    html.Div(id='live-update-text'),
    html.Hr(),
    ], style={"padding":"10px"})

@app.callback([dash.dependencies.Output('live-update-text', 'children')],
              inputs=[dash.dependencies.Input(i, 'value') for i in ["expression"]])
def get_graphs(expression):
    style = dict()
    try:
        expr = to_ak_expr(expression)
        style["color"] = "#228B22"
        return [html.Div([
                    html.Code(expr, style=style),
                    html.Br(), html.Br(),
                    html.Code("Result: " + str(df.draw(expression, to_array=True)), style=style),
                    ])]
    except SyntaxError as e:
        style["color"] = "#ff0000"
        style["white-space"] = "pre-wrap"
        tb = traceback.format_exc()
        expr_line, arrow_line, error_line = tb.splitlines()[-3:]
        return [html.Div([
            html.Code("SyntaxError:", style=style), html.Br(),
            html.Code(expr_line, style=style), html.Br(),
            html.Code(arrow_line, style=style), html.Br(),
            ])]
    except KeyError as e:
        style["color"] = "#ff0000"
        style["white-space"] = "pre-wrap"
        tb = traceback.format_exc()
        return [html.Div([
            html.Code(tb.splitlines()[-1], style=style), html.Br(),
            ])]
    except:
        tb = traceback.format_exc()
        out = tb
        return [html.Code(out, style=style)]

if __name__ == '__main__':
    app.run_server(debug=True)
