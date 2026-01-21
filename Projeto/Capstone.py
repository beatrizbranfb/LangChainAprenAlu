from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()
loader = PyPDFDirectoryLoader("documentos_curso/", silent_errors=True)
docs = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ".", ",", ""]
)

chunks = text_splitter.split_documents(docs)

print(f"Total de chunks criados: {len(chunks)}")

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model)

print("Banco de dados criado com sucesso!")

from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5
vector_retriever = vectorstore.as_retriever(search_kwargs={"k":5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

print(f"[Recuperador de Busca Híbrida Configurado]")

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=ensemble_retriever,
    memory=memory,
    verbose=False,
    return_source_documents=True,
    get_chat_history=lambda h : "\n".join([message.content for message in h])
)

resposta1 = qa_chain.invoke({"question":"O que é chunking adaptativo?"})
print(resposta1['answer'])

resposta2 = qa_chain.invoke({"question":"E quais as principais estratégias?"})
print(resposta2['answer'])

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# 1. Crie um conjunto de dados para avaliação
perguntas = [
    "O que é RAG e qual problema ele soluciona?",
    "Quais os componentes essenciais do RAG?",
    "Qual a diferença entre busca lexical e semântica?",
    "O que mede a métrica faithfulness do RAGAS?"
]

respostas_puro = [
    "RAG (Retrieval-Augmented Generation) é uma arquitetura que combina um motor de busca para recuperar informações com um L",
    "Os componentes essenciais são: Embeddings, Banco de Dados Vetorial, chunking e um Modelo de Linguagem (LLM).",
    "Busca lexical (como BM25) encontra correspondências exatas de termos, enquanto a busca semântica captura o significado e ",
    "A métrica Faithfulness mede se a resposta gerada é suportada e factualmente consistente com os documentos recuperados, e"
]

# 2. Gere as respostas e contextos com a nossa cadeia
respostas_geradas = []
contextos_recuperados = []
for question in perguntas:
    result = qa_chain.invoke({"question": question})
    respostas_geradas.append(result['answer'])
    contextos_recuperados.append([doc.page_content for doc in result['source_documents']])

# 3. Crie o dataset no formato esperado pelo RAGAS
dataset_dict = {
    'question': perguntas,
    'answer': respostas_geradas,
    'contexts': contextos_recuperados,
    'ground_truth': respostas_puro
}

dataset = Dataset.from_dict(dataset_dict)

# 4. Execute a avaliação, agora usando os modelos do Google
evaluation_result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest"),
    embeddings=embeddings_model
)

# 5. Analise os resultados
df_resultados = evaluation_result.to_pandas()
print("\nResultados da Avaliação com RAGAS:")
print(df_resultados)