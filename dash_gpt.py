#!/usr/bin/env python3
# coding: utf-8

# importa o módulo base64 para trabalhar com codificação e decodificação de dados
import base64
# importa o módulo os para interagir com o sistema operacional
import os
# importa o objeto ConfigParser do modulo configparser para gerenciar o arquivo de configuração
from configparser import ConfigParser

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
#   por similaridade em textos e carregar um modelo de predição de linguagem natural
from llama_index import (GPTSimpleVectorIndex, LLMPredictor, ServiceContext,
                         download_loader)

# registra a atual versão do app
version = "4.0.0"
# obter o caminho absoluto do diretório atual
path = os.path.dirname(os.path.abspath(__file__))
# define o caminho para o diretório onde os arquivos serão salvos
folder = f"{path}/dash_gpt_pdf/"
# criacao do objeto "config" para gerenciar o trabalho com o arquivo de configuração .ini
config = ConfigParser()
# define o nome do arquivo .ini que será utilizado para armazenar a chave de API do OpenAI
config_ini = "dash_gpt.ini"
# define o nome do arquivo JSON que será utilizado para indexar os modelos de linguagem treinados
index_json = "dash_gpt.json"
# verifica se o diretório existe e, se não existir, cria o diretório
if not os.path.exists(folder):
    os.makedirs(folder)

# função de criação do índice de vetor simples a partir dos documentos PDF carregados
def create_index(key_test=False):
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
        # criação do objeto "LLMPredictor" com o modelo gpt-3.5-turbo do OpenAI
        llm_predictor = LLMPredictor(llm=ChatOpenAI(model_name="gpt-3.5-turbo"))
        # criação do objeto "ServiceContext" com as configurações padrão e com o objeto "LLMPredictor"
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=4096)
        # criação do índice a partir dos documentos carregados, utilizando o objeto "GPTSimpleVectorIndex"
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
        # salvamento do índice em disco
        index.save_to_disk(index_json)
        # verifica ou não a chave da API em função do parâmetro "key_test"
        if key_test:
            # verifica se a chave da API do OpenAI informada está ok
            index.query("DAN", mode=llama_index.QueryMode.EMBEDDING)
        # retorna verdadeiro
        return True
    # se houver uma exceção (erro) em qualquer parte do bloco
    except:
        # retorna falso
        return False

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
def answer(enter, click, query, out):
    # verifica se o arquivo "config.py" já existe
    if not os.path.exists(config_ini):
        # se não possui qualquer histórico
        if not out:
            # solicita que o usuário digite a chave de API do OpenAI
            out.append(html.Dd("Não encontrei a chave da API do OpenAI. Por favor, digite a chave e pressione ENVIAR"))
        # se possui alguma informação digitada pelo usuário
        if query:
            # define a variável de ambiente para a chave da API do OpenAI
            os.environ["OPENAI_API_KEY"] = query
            # executa a função create_index com a chave da API do OpenAI informada
            #   se a chave estevier ok, returna verdadeiro e salva no arquivo "config.py"
            #   caso contrário retorna falso e solicita novamente a chave
            if create_index(True):
                # cria a seção "OPENAI"
                config.add_section("OPENAI")
                # cria na seção "OPENAI" o parâmetro "OPENAI_API_KEY" com o valor da variável "query"
                config.set("OPENAI", "OPENAI_API_KEY", query)
                # abre (e se não existe, cria) o arquivo "dash_gpt.ini" no modo de escrita
                with open(config_ini, "w") as config_file:
                    # escreve o conteúdo da variável "config" no arquivo "dash_gpt.ini"
                    config.write(config_file)
                # define o conteúdo da variável "query" para uma string vazia
                query = ""
                # define o conteúdo da variável "out" para uma lista vazia
                out = []
                # adiciona um texto inicial ao conteúdo da variável "out"
                out.append(html.Dd("Seja Bem-Vindo!"))
            else:
                # informa o usuário que a chave informada é inválida e solicita-a novamente
                out.append(html.Dd("A chave da API do OpenAI informada é inválida. Por favor, digite uma chave válida e pressione ENVIAR"))
    # se existe o arquivo de configuração e não existe o arquivo de index
    if os.path.exists(config_ini) and not os.path.exists(index_json):
        # carrega a chave de acesso da API do OpenAI da variável OPENAI_API_KEY do arquivo de configuração
        config.read(config_ini)
        # define a variável de ambiente para a chave da API do OpenAI
        os.environ["OPENAI_API_KEY"] = config.get("OPENAI", "OPENAI_API_KEY")
        # executa a função que cria o índice
        create_index()
    # se existe alguma mensagem do usuário e existem os arquivos de configuração e de index
    if query and os.path.exists(config_ini) and os.path.exists(index_json):
        # carrega a chave de acesso da API do OpenAI da variável OPENAI_API_KEY do arquivo de configuração
        config.read(config_ini)
        # define a variável de ambiente para a chave da API do OpenAI
        os.environ["OPENAI_API_KEY"] = config.get("OPENAI", "OPENAI_API_KEY")
        # carrega o índice salvo anteriormente utilizando o método "load_from_disk"
        #   da classe "GPTSimpleVectorIndex" e atribui o resultado à variável "index"
        index = GPTSimpleVectorIndex.load_from_disk(index_json)
        # obtém uma resposta do índice criado
        response = str(index.query(query, mode=llama_index.QueryMode.EMBEDDING))
        # adiciona a pergunta feita ao conteúdo da variável "out"
        out.append(html.Dt(query))
        # adiciona a resposta dada ao conteúdo da variável "out"
        out.append(html.Dd(response))
    # retorna o conteúdo dos widgets dash
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
