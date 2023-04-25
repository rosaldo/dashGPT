#!/usr/bin/env python3
# coding: utf-8

import base64
import os

import dash_bootstrap_components as dbc
import llama_index
from dash import Dash, Input, Output, State, dcc, html
from flask import Flask
from langchain.chat_models import ChatOpenAI
from llama_index import (GPTSimpleVectorIndex, LLMPredictor, ServiceContext,
                         download_loader)

from config import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
path = os.path.dirname(os.path.abspath(__file__))
folder = f"{path}/dash_gpt_pdf/"
index_json = "dash_gpt.json"
if not os.path.exists(folder):
    os.makedirs(folder)

def create_index():
    pdf_reader = download_loader("PDFReader")
    files = os.listdir(folder)
    documents = []
    for file in files:
        if file.endswith(".pdf"):
            pdf = f"{path}/dash_gpt_pdf/{file}"
            document = pdf_reader().load_data(pdf)
            documents.append(document[0])
    llm_predictor = LLMPredictor(llm=ChatOpenAI(model_name="gpt-4-32k"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=4096)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    index.save_to_disk(index_json)

create_index()
index = GPTSimpleVectorIndex.load_from_disk(index_json)

server = Flask(__name__)
app = Dash(
    __name__,
    server=server,
    title="dashGPT",
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.COSMO],
    meta_tags=[
        {"charset": "utf-8"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1, shrink-to-fit=no"},
    ],
)
app.layout = dbc.Container(
    children=[
        dcc.Store(id="out", data=[]),
        dbc.Row(html.H1([html.I("dash"), html.B("GPT")]), style={"text-align":"center"}),
        dbc.Row(
            html.Div(
                html.Div(id="output"),
                style={"height":"100%", "overflowY":"auto", "display":"flex", "flex-direction":"column-reverse"}),
            style={"width":"100%", "height":"calc(100vh - 200px)", "margin-top":"25px", "margin-bottom":"25px"}),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Upload(
                        id="uppdf",
                        accept=".pdf,.PDF",
                        multiple=True,
                        children=dbc.Button("PDFs"),
                    ),
                    width=1,
                ),
                dbc.Col(
                    dbc.Row(
                        [
                            dbc.Input(id="input", placeholder="digite uma pergunta", autofocus=True, style={"width":"calc(100% - 85px)", "margin-right":"5px"}),
                            dbc.Button("Enviar", id="bt_send", style={"width":"80px"})
                        ]
                    ),
                    width=11,
                )
            ],
        )
    ]
)

@app.callback(
    [
        Output("uppdf", "filename"),
        Output("uppdf", "contents"),
    ],
    [
        Input("uppdf", "filename"),
    ],
    [
        State("uppdf", "contents"),
    ]
)
def uppdf(files, contents):
    if files and contents:
        for file, content in zip(files, contents):
            content_decoded = base64.b64decode(content[28:])
            with open(os.path.join(folder, file), "wb") as pdf:
                pdf.write(content_decoded)
        create_index()
    return ["", ""]

@app.callback(
    [
        Output("output", "children"),
        Output("input", "value"),
        Output("out", "data"),
    ],
    [
        Input("input", "n_submit"),
        Input("bt_send", "n_clicks"),
    ],
    [
        State("input", "value"),
        State("out", "data"),
    ]
)
def answer(enter, click, input, out):
    if input:
        response = str(index.query(input, mode=llama_index.QueryMode.EMBEDDING))
        if not response:
            response = "Sem informações. Envie um PDF com os dados para que eu pesquise."
        out.append(html.Dt(input))
        out.append(html.Dd(response))
    return [out, "", out]

self_name = os.path.basename(__file__)[:-3]
if len(os.sys.argv) == 1:
    app.run(host="127.0.0.1", port="8888", debug=True)
elif len(os.sys.argv) == 2:
    host = os.sys.argv[1]
    os.system(f"gunicorn {self_name}:server -b {host}:8888 --reload --timeout 120")
elif len(os.sys.argv) == 3:
    host = os.sys.argv[1]
    port = int(os.sys.argv[2])
    os.system(f"gunicorn {self_name}:server -b {host}:{port} --reload --timeout 120")
