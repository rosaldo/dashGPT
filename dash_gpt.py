#!/usr/bin/env python3
# coding: utf-8

# importa o módulo base64 para trabalhar com codificação e decodificação de dados
import base64
# importa o módulo os para interagir com o sistema operacional
import os

# importa o Dash Bootstrap Components para trabalhar com elementos visuais do Dash
import dash_bootstrap_components as dbc
# importa o módulo llama_index para indexação e busca em modelos de linguagem
import llama_index
# importa os componentes essenciais do Dash para criar a aplicação
from dash import Dash, Input, Output, State, dcc, html
# importa o Flask para criar o servidor web
from flask import Flask
# importa a classe ChatOpenAI do módulo chat_models do pacote langchain
from langchain.chat_models import ChatOpenAI
# importa classes e função que são usadas para criar um indexador de dados para realizar buscas
# por similaridade em textos e carregar um modelo de predição de linguagem natural
from llama_index import GPTSimpleVectorIndex, LLMPredictor, ServiceContext, download_loader

# verifica se o arquivo "config.py" já existe
if not os.path.exists("config.py"):
    # solicita que o usuário digite a chave de API do OpenAI e armazena na variável "api_key"
    api_key = input("Digite a chave de API do OpenAI: ")
    # abre (e se não existe, cria) o arquivo "config.py" no modo de escrita e armazena na variável "conf"
    conf = open("config.py", "w")
    # escreve a chave de API no arquivo "config.py"
    conf.write(f"""OPENAI_API_KEY = "{api_key}"\n""")
    # fecha o arquivo "config.py"
    conf.close()

# importa a chave de acesso da API do OpenAI da variável OPENAI_API_KEY do módulo config
from config import OPENAI_API_KEY

# define a variável de ambiente para a chave da API do OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# obter o caminho absoluto do diretório atual
path = os.path.dirname(os.path.abspath(__file__))
# define o caminho para o diretório onde os arquivos serão salvos
folder = f"{path}/dash_gpt_pdf/"
# define o nome do arquivo JSON que será utilizado para indexar os modelos de linguagem treinados
index_json = "dash_gpt.json"
# verifica se o diretório existe e, se não existir, cria o diretório
if not os.path.exists(folder):
    os.makedirs(folder)

# função de criação do índice de vetor simples a partir dos documentos PDF carregados
def create_index():
    # importação do loader "PDFReader" do pacote llama_index
    pdf_reader = download_loader("PDFReader")
    # listagem dos arquivos do diretório definido
    files = os.listdir(folder)
    # criação de uma lista vazia para armazenar os documentos
    documents = []
    # laço que percorre os arquivos da listagem armazenada em "files"
    for file in files:
        # verifica se o arquivo possui a extensão ".pdf"
        if file.endswith(".pdf"):
            # armazena o caminho completo do arquivo
            pdf = f"{path}/dash_gpt_pdf/{file}"
            # carregamento do documento PDF
            document = pdf_reader().load_data(pdf)
            # adição do documento à lista de documentos
            documents.append(document[0])
    # tenta executar o bloco abaixo
    try:
        # criação do objeto "LLMPredictor" com o modelo GPT-4-32K do OpenAI
        llm_predictor = LLMPredictor(llm=ChatOpenAI(model_name="gpt-4-32k"))
        # criação do objeto "ServiceContext" com as configurações padrão e com o objeto "LLMPredictor"
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=4096)
        # criação do índice a partir dos documentos carregados, utilizando o objeto "GPTSimpleVectorIndex"
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
        # salvamento do índice em disco
        index.save_to_disk(index_json)
    # se houver uma exceção (erro) em qualquer parte do bloco
    except:
        # termina com um código de saída 1
        os.sys.exit(1)

# executa a função "create_index" definida acima
create_index()
# carrega o índice salvo anteriormente utilizando o método "load_from_disk"
# da classe "GPTSimpleVectorIndex" e atribui o resultado à variável "index"
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
