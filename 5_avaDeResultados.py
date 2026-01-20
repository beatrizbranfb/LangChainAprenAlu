# Importações necessárias
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from datasets import Dataset

# Criação de documentos de exemplo sobre IA e tecnologia
documentos_conhecimento = [
    """Inteligência Artificial (IA) é um campo da ciência da computação que se concentra
na criação de sistemas capazes de realizar tarefas que normalmente requerem inteligência humana.
Isso inclui aprendizado, raciocínio, percepção e tomada de decisões. A IA pode ser classificada
em IA fraca (específica para tarefas) e IA forte (inteligência geral).""",
    """Machine Learning é uma subárea da IA que permite que computadores aprendam e melhorem
automaticamente através da experiência, sem serem explicitamente programados.
Os algoritmos de ML identificam padrões em dados e fazem previsões. Existem três tipos principais:
aprendizado supervisionado, não supervisionado e por reforço.""",
    """Deep Learning é uma técnica de machine learning baseada em redes neurais artificiais
com múltiplas camadas. É especialmente eficaz para tarefas como reconhecimento de imagem,
processamento de linguagem natural e reconhecimento de voz. As redes neurais profundas
podem ter centenas de camadas e milhões de parâmetros.""",
    """RAG (Retrieval-Augmented Generation) é uma técnica que combina recuperação de informações
com geração de texto. Permite que modelos de linguagem acessem conhecimento externo
para gerar respostas mais precisas e atualizadas. O processo envolve buscar documentos
relevantes e usar essas informações para gerar a resposta final.""",
    """Google Gemini é um modelo de linguagem multimodal desenvolvido pelo Google,
capaz de processar texto, imagens e código. Oferece capacidades avançadas de
raciocínio e compreensão contextual. O Gemini vem em diferentes versões:
Nano, Pro e Ultra, cada uma otimizada para diferentes casos de uso.""",
    """LangChain é um framework para desenvolvimento de aplicações com modelos de linguagem.
Facilita a criação de cadeias complexas, gerenciamento de memória e integração
com diferentes fontes de dados. Oferece componentes modulares para construir
aplicações robustas de IA conversacional."""
]

# Conversão para objetos Document
docs = [Document(page_content=doc) for doc in documentos_conhecimento]
print(f"✅ Criados {len(docs)} documentos de conhecimento")

# Criação de dataset de teste para avaliação RAGAS
dados_teste = {
    'question': [
        "O que é Inteligência Artificial?",
        "Como funciona o Machine Learning?",
        "Quais são as aplicações do Deep Learning?",
        "O que é RAG e como funciona?",
        "Quais são as características do Google Gemini?"
    ],
    'ground_truth': [
        "Inteligência Artificial é um campo da ciência da computação focado na criação de sistemas que realizam tarefas que requerem inteligência humana, como aprendizado, raciocínio e percepção.",
        "Machine Learning permite que computadores aprendam automaticamente através da experiência, identificando padrões em dados para fazer previsões.",
        "Deep Learning é eficaz para reconhecimento de imagem, processamento de linguagem natural e reconhecimento de voz, usando redes neurais profundas.",
        "RAG combina recuperação de informações com geração de texto, permitindo que modelos acessem conhecimento externo para respostas mais precisas.",
        "Google Gemini é um modelo multimodal que processa texto, imagens e código, oferecendo capacidades avançadas de raciocínio e vem em versões como Nano, Pro e Ultra."
    ]
}
print("✅ Dataset de teste criado!")
print(f"✅ {len(dados_teste['question'])} perguntas de teste preparadas")

# Inicializacao dos embeddings do Google Gemini
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=load_dotenv("GOOGLE_API_KEY")
)

# Criação do vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db_avaliacao"
)

# Criação do modelo Google Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.3,
    google_api_key=load_dotenv("GOOGLE_API_KEY"),
    convert_system_message_to_human=True
)

# Configuracao da memória
memory= ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True,
    output_key='answer'
)

# Criação da cadeira RAG
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,
    verbose=False
)

def executar_rag_e_coletar_dados(perguntas):
    """Executa o sistema RAG e coleta dados para avaliação"""
    
    resultados = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': []
    }
    
    for i, pergunta in enumerate(perguntas):
        print(f"Processando pergunta {i+1}/{len(perguntas)}: {pergunta}")
        
        try:
            # Executar RAG
            resultado = rag_chain({"question": pergunta})
            
            # Extrair contextos dos documentos fonte
            contextos = [doc.page_content for doc in resultado['source_documents']]
            
            # Armazenar resultados
            resultados['question'].append(pergunta)
            resultados['answer'].append(resultado['answer'])
            resultados['contexts'].append(contextos)
            resultados['ground_truth'].append(dados_teste['ground_truth'][i])
            
            print(f"Resposta gerada: {resultado['answer'][:100]}...")
            
        except Exception as e:
            print(f"Erro ao processar pergunta: {str(e)}")
            continue
            
    return resultados

# Executar coleta de dados
print("Iniciando coleta de dados para avaliação...")
dados_avaliacao = executar_rag_e_coletar_dados(dados_teste['question'])
print(f"\nColeta concluída! {len(dados_avaliacao['question'])} exemplos coletados")

# Preparar dataset para RAGAS
dataset_ragas = Dataset.from_dict(dados_avaliacao)
print("Dataset preparado para avaliação RAGAS:")
print(f" - {len(dataset_ragas)} exemplos")
print(f" - Colunas: {list(dataset_ragas.column_names)}")

# Configurar métricas RAGAS
metricas_ragas = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
]

print("\n Métricas RAGAS configuradas:")
for metrica in metricas_ragas:
    print(f" - {metrica.name}")

# Executar avaliação RAGAS
print("Iniciando avaliação com RAGAS...")
print("Isso pode levar alguns minutos...")

# Configurar LLM para RAGAS (usando Gemini)
resultado_ragas = evaluate(
    dataset_ragas,
    metrics=metricas_ragas,
    llm=llm,
    embeddings=embeddings
)

print("\n Avaliação RAGAS concluída!")

# Exibir resultados
print("\n === RESULTADOS DA AVALIAÇÃO RAGAS ===")

# Access individual metrics from the EvaluationResult object
print(f"faithfulness: {resultado_ragas['faithfulness'][0]:.4f}")
print(f"answer_relevancy: {resultado_ragas['answer_relevancy'][0]:.4f}")
print(f"context_precision: {resultado_ragas['context_precision'][0]:.4f}")
print(f"context_recall: {resultado_ragas['context_recall'][0]:.4f}")

def analisar_metricas_ragas(resultados):
    """Analisa e interpreta um objeto EvaluationResult (ou dict) do RAGAS,
    imprimindo cada métrica com uma leitura qualitativa."""
    
    # 1) Converter o objeto EvaluationResult para dicionário simples
    if hasattr(resultados, "to_dict"): # RAGAS >= 0.0.12
        resultados = resultados.to_dict()
    elif hasattr(resultados, "_scores_dict"): # RAGAS 0.0.10 - 0.0.11
        resultados = resultados._scores_dict
    elif not isinstance(resultados, dict): # Qualquer Mapping restante
        resultados = dict(resultados) # Conversion fallback
    
    # 2) Helper seguro para obter o score como escalar
    def pegar_score(chave: str, default: float = 0.0) -> float:
        valor = resultados.get(chave, default)
        # Alguns back-ends devolvem lista ou ScoreList -> pega primeiro item
        if isinstance(valor, (list, tuple)):
            return float(valor[0])
        return float(valor)
        
    print("\n === ANÁLISE DETALHADA DAS MÉTRICAS RAGAS ===\n")
    
    # 3) Métricas esperadas (algumas instalações usam context_precision
    # no lugar de context_relevancy - cobrimos ambos)
    metricas = {
        "faithfulness": "Faithfulness (Factualidade)",
        "answer_relevancy": "Answer Relevancy (Relevância da Resposta)",
        "context_relevancy": "Context Relevancy (Relevância do Contexto)",
        "context_precision": "Context Precision (Precisão do Contexto)",
        "context_recall": "Context Recall (Recall do Contexto)",
    }
    
    scores_validos = []
    for chave, nome_legivel in metricas.items():
        if chave not in resultados: # pula métricas ausentes
            continue
        score = pegar_score(chave)
        scores_validos.append(score)
        
        # Exibe valor numérico
        print(f"**{nome_legivel}: {score:.4f}**")
        
        # Interpretação qualitativa
        if score >= 0.80:
            print(" Excelente!")
        elif score >= 0.60:
            print(" Moderado.")
        else:
            print(" Baixo.")
            
    # 4) Score geral (média simples dos scores presentes)
    if scores_validos:
        score_geral = float(np.mean(scores_validos))
        print(f"\n**Score Geral: {score_geral:.4f}**")
        if score_geral >= 0.80:
            print(" Sistema RAG com performance excelente!")
        elif score_geral >= 0.60:
            print(" Sistema RAG com performance boa, mas com espaço para melhorias")
        else:
            print(" Sistema RAG precisa de otimizações significativas")
    else:
        print("Nenhuma métrica disponível para análise.")

analisar_metricas_ragas(resultado_ragas)