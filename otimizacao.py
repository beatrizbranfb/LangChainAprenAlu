import os 
import time
import numpy as np
import keySecure
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import time  

key = keySecure.get_google_api_key(self=None)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

texto_teste = ["O que o TCU faz?,"
"Qual o nome do presidente do TCU?,"
"Quantos ministros tem o TCU?"] 

start_time = time.time()
embeddings_gemini = embeddings.embed_documents(texto_teste)
end_time = time.time()
print(f"tempo de geração do embedding com Gemini: {end_time - start_time} segundos")
print(f"Embedding Gemini: {len(embeddings_gemini[0])}")

#minilm
minilm_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
start_time = time.time()
minilm_embeddings = embeddings.embed_documents(texto_teste)
end_time = time.time()
print(f"tempo de geração: {end_time - start_time} segundos")
print(f"Embedding Gemini: {len(minilm_embeddings[0])}")