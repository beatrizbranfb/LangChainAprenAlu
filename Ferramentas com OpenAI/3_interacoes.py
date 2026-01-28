import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
api_key = os.getenv("OpenAi_API_Key")

modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)

prompt_sugestao = ChatPromptTemplate.from_messages([
    ("system", "Você é um guia de viagem especializado em destinos brasileiros. Apresente-se como Sr. Passeios"),
    ("placeholder", "{historico}"),
    ("human", "{query}")
])

chain = prompt_sugestao | modelo | StrOutputParser()

memoria = {}
sessao = "aula_langchain"

def historico_por_sessao(sessao_id: str):
    if sessao_id not in memoria:
        memoria[sessao_id] = InMemoryChatMessageHistory()
    return memoria[sessao_id]

lista_perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

cadeia_com_historico = RunnableWithMessageHistory(
    runnable=chain, 
    get_message_history=lambda _: historico_por_sessao(sessao),
    input_messages_key="query",
    history_messages_key="historico"
)

for uma_pergunta in lista_perguntas:
    resposta = cadeia_com_historico.invoke(
        {
            "query": uma_pergunta
        },
        config={"session_id": sessao}   )
    print("Usuário: ", uma_pergunta)
    print("IA: ", resposta.content, "\n")

