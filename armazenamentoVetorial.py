import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings #Embeddings transformam dados complexos em vetores 
from langchain_core.documents import Document
import keySecure #importa o arquivo de chaves
from langchain_community.vectorstores import FAISS #Facebook AI Similarity Search
import faiss 

#Configuração da chave de API do Google a partir do arquivo keySecure.py
key = keySecure.get_google_api_key(self=None)

d = 768
index_hns2 = faiss.IndexHNSWFlat(d, 32) #32 é o número de conexões por nó

embeddings = GoogleGenerativeAIEmbeddings(model="models/embeddings-001")

#faiss_db = FAISS.pasta(nome_documento, embeddings) 

#pinecone 
