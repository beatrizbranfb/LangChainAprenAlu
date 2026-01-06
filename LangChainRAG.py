import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings #Embeddings transformam dados complexos em vetores 
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import faiss
import keySecure #importa o arquivo de chaves

#Configuração da chave de API do Google a partir do arquivo keySecure.py
key = keySecure.get_google_api_key(self=None)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 0)

 #exemplo de prompt tradicional sem RAG

pergunta = "O que o TCU faz?"

prompt_tradicional = ChatPromptTemplate.from_template(
    "Responda a seguinte pergunta: {pergunta}"
)

chain_tradicional = prompt_tradicional | llm
resposta_tradicional = chain_tradicional.invoke({"pergunta":pergunta})
print(resposta_tradicional.content)

#Embeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embeddings-001")