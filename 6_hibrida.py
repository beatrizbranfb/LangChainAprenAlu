import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain.retrievers import EnsembleRetriever
import uuid
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Documentos de exemplo para os testes
docs = [ # Lista de documentos de conhecimento
    Document(page_content="O erro HTTP 404 Not Found indica que o servidor não conseguiu encontrar o recurso solicitado. Verifique a URL.", metadata={"source":"doc_erros_http"}),
    Document(page_content="O protocolo SSH (Secure Shell) é usado para acesso remoto seguro a servidores. A porta padrão é a 22.", metadata={"source":"doc_protocolos_rede"}),
    Document(page_content="Para listar contêineres Docker, use o comando 'docker ps'. O erro 'Cannot connect to the Docker daemon' significa que o serviço do Docker não está em execução.", metadata={"source":"doc_docker"}),
    Document(page_content="A política de férias da empresa permite 30 dias de descanso remunerado por ano. Para solicitar, acesse o portal de RH e preencha o formulário 'FRM-01-FERIAS'.", metadata={"source":"doc_politicas_rh"})
]

#implementação da busca híbrida
from langchain.retrievers import BM25Retriever, EnsembleRetriecer
from langchain_community.vectorstores import FAISS

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 2 #ajuste do peso para fusão dos resultados

# Retriever Vetorial (Denso)
faiss_vectorstore = FAISS.from_documents(docs, embeddings)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

# Ensemble Retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)

query_keyword = "Como peço férias usando o formulário FRM-01-FERIAS?"

print(f"--- Buscando por '{query_keyword}' --- \n")

# Busca apenas vetorial
print("--- Resultado da busca vetorial (FAISS) ---")
doc_faiss = faiss_retriever.invoke(query_keyword)
for doc in doc_faiss:
    print(f"-- {doc.page_content}")

# Busca Híbrida
print("\n--- Resultado da busca híbrida (EnsembleRetriever) ---")
doc_ensemble = ensemble_retriever.invoke(query_keyword)
for doc in doc_ensemble:
    print(f"-- {doc.page_content}")


# --- Teste de Recuperação ---
query_keyword = "Como peço férias usando o formulário FRM-01-FERIAS?"

def show_results(titulo, docs, termo="FRM-01-FERIAS"):
    print(f"\n--- {titulo} ---")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "sem_source")
        print(f'{i} src="{src}" | {d.page_content[:120]}...') # desconente se quiser ver o início do texto

print(f"--- Buscando por: '{query_keyword}' ---")

# Busca vetorial (FAISS)
docs_faiss = faiss_retriever.invoke(query_keyword)
show_results("Resultados da busca vetorial (FAISS)", docs_faiss)

# Busca híbrida (BM25 + FAISS via EnsembleRetriever)
docs_ensemble = ensemble_retriever.invoke(query_keyword)
show_results("Resultados da busca híbrida (EnsembleRetriever)", docs_ensemble)

# Análise automática simples
faiss_top = docs_faiss[0].metadata.get("source")
ensemble_top = docs_ensemble[0].metadata.get("source")

# Documento longo para o exemplo
doc_longo = [
    Document(
        page_content="""
Introdução à Segurança Cibernética (2024)...

Uma das técnicas de ataque mais comuns é o Phishing...

Conclusão: Manter-se atualizado... A autenticação de dois fatores (2FA) deve ser obrigatória.
""",
        metadata={"source": "guia_seguranca_ciber.pdf", "ano": 2024}
    )
]

# 1. Splitter para dividir o documento em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
doc_chunks = text_splitter.split_documents(doc_longo)

# 2. Gerador de Resumos
def generate_summaries(docs, llm_model):
    """Gera resumos para uma lista de documentos."""
    prompt = ChatPromptTemplate.from_template(
        "Resuma o seguinte documento em uma frase: {documento}"
    )
    chain = prompt | llm_model
    summaries = chain.batch([{"documento": doc.page_content} for doc in docs])
    return [s.content for s in summaries]

doc_ids = [str(uuid.uuid4()) for _ in doc_chunks] # IDs únicos para cada chunk
summary_chunks = generate_summaries(doc_chunks, llm)

store = InMemoryStore()
store.mset(list(zip(doc_ids, doc_chunks)))

# Cria vetor-store para os resumos, carregando o id de origem do metadado
summary_vectorstore = FAISS.from_texts(
    summary_chunks,
    embeddings,
    metadatas=[{"doc_id": doc_ids[i]} for i in range(len(summary_chunks))]
)

multi_vector_retriever = MultiVectorRetriever(
    vectorstore=summary_vectorstore,
    docstore=store,
    id_key="doc_id",
    search_kwargs={'k': 1}
)

# Teste de Recuperacao
query_resumo = "qual a principal defesa contra ataques cibernéticos?"
retrieved_docs = multi_vector_retriever.invoke(query_resumo)
print(f"--- Buscando por: '{query_resumo}' ---\n")
print(f"--- Documento Original Recuperado via Resumo (MultiVectorRetriever) ---")

if retrieved_docs:
    print(retrieved_docs[0].page_content)
else:
    print("No documents were retrieved.")