import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS 
import faiss #Facebook AI Similarity Search
from langchain_community.vectorstores import Chroma

load_dotenv() #Puxar a chave do arquivo dotenv
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

#embeddings.embed_query("Texto de exemplo para gerar embedding")

Document(
    page_content="Texto que descreve o conteúdo informativo",
    metadata={
        "chave1": "valor",
        "chave2": "valor",
    }
)

documentos_empresa = [
    Document(
        page_content="Política de férias: Funcionários têm direito a 30 dias de férias após 12 meses. A solicitação deve ser feita com 30 dias de antecedência.",
        metadata={"tipo": "política", "departamento": "RH", "ano": 2024, "id_doc": "doc001"}
    ),
    Document(
        page_content="Processo de reembolso de despesas: Envie a nota fiscal pelo portal financeiro. O reembolso ocorre em até 5 dias úteis.",
        metadata={"tipo": "processo", "departamento": "Financeiro", "ano": 2023, "id_doc": "doc002"}
    ),
    Document(
        page_content="Guia de TI: Para configurar a VPN, acesse vpn.nossaempresa.com e siga as instruções para seu sistema operacional.",
        metadata={"tipo": "tutorial", "departamento": "TI", "ano": 2024, "id_doc": "doc003"}
    ),
    Document(
        page_content="Código de Ética e Conduta: Valorizamos o respeito, a integridade e a colaboração. Casos de assédio não serão tolerados.",
        metadata={"tipo": "política", "departamento": "RH", "ano": 2022, "id_doc": "doc004"}
    )
]

faiss_db = FAISS.from_documents(documentos_empresa, embeddings)

pergunta = "Quais são as políticas de férias da empresa?"
resultados = faiss_db.similarity_search(pergunta, k=2)
d =768
index_hnsw = faiss.IndexHNSWFlat(d, 32) #32 é o número de conexões por nó
faiss_db = FAISS.from_documents(documentos_empresa, embeddings)
resultados = faiss_db.similarity_search(pergunta, k=2)
print(f"\n Pergunta: {pergunta}")
print("\n Documentos mais relevantes (FAISS):")
for i, doc in enumerate(resultados):
    print(f"\n Documento {i+1}:\n{doc.page_content}\nMetadados: {doc.metadata}")

chroma_db = Chroma.from_documents(
    documents = documentos_empresa,
    embedding = embeddings
)

resultados = chroma_db.similarity_search(pergunta, k=2)
for doc in resultados:
    print(f"\n {doc.page_content}")

pergunta_rh = "Quais as regras da empresa?"

resultados_filtrados = chroma_db.similarity_search(
    pergunta_rh,
    k=2,
    filter={"$and": [{"departamento": "RH"}, {"tipo": "política"}]}
)

for doc in resultados_filtrados:
    print(f"\n {doc.page_content}")
    print(f"Metadados: {doc.metadata}")