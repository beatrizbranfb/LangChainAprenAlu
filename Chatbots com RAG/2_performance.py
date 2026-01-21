import os
import time
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
 
load_dotenv();

textos_teste = [
    "Qual é a política de férias da nossa empresa?",
    "Preciso de um relatório de despesas de viagem.",
    "Como configuro o acesso à rede privada virtual (VPN)?",
    "Onde encontro o código de conduta da organização?",
    "Quero entender o processo de avaliação de performance."
]

#google
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

start_time = time.time()
embeddings_gemini = gemini_embeddings.embed_documents(textos_teste)
end_time = time.time()

print(f"Tempo de processamento: {end_time - start_time} segundos")
print(f"Dimensões do vetor: {len(embeddings_gemini[0])}")

#all-MiniLM-L6-v2

minilm_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
start_time = time.time()
embeddings_minilm = minilm_embeddings.embed_documents(textos_teste)
end_time = time.time()

print(f"Tempo de processamento: {end_time - start_time} segundos")
print(f"Dimensões do vetor: {len(embeddings_minilm[0])}")

#BGE large

bge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

start_time = time.time()
embeddings_bge = bge_embeddings.embed_documents(textos_teste)
end_time = time.time()

print(f"Tempo de processamento: {end_time - start_time} segundos")
print(f"Dimensões do vetor: {len(embeddings_bge[0])}")

#Pergunta
pergunta = "Quero tirar uns dias de folga do trabalho."

emb_pergunta_gemini = gemini_embeddings.embed_query(pergunta)
emb_pergunta_minilm = minilm_embeddings.embed_query(pergunta)
emb_pergunta_bge = bge_embeddings.embed_query(pergunta)

modelos = {
    "Gemini": (emb_pergunta_gemini, embeddings_gemini),
    "MiniLM": (emb_pergunta_minilm, embeddings_minilm),
    "BGE-large": (emb_pergunta_bge, embeddings_bge)
}

for nome, (emb_q, emb_docs) in modelos.items():
    similaridades = cosine_similarity([emb_q], emb_docs)[0]
    doc_e_similaridade = sorted(
        zip(textos_teste, similaridades), key=lambda x: x[1], reverse=True
    )
    print(f"--- Ranking para o modelo {nome} ---")
    for i, (doc, sim) in enumerate(doc_e_similaridade[:3], 1):
        print(f"{i}. (Score: {sim:.3f}) {doc}")
    print()