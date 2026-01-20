# [] Importações necessárias
import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import google.generativeai as genai
import warnings

load_dotenv()  
warnings.filterwarnings('ignore')

# Criação de documentos de exemplo sobre IA e Machine Learning
documentos_exemplo = [
    """Inteligência Artificial (IA) é um campo da ciência da computação que se concentra
na criação de sistemas capazes de realizar tarefas que normalmente requerem inteligência humana.
Isso inclui aprendizado, raciocínio, percepção e tomada de decisões.""",

    """Machine Learning é uma subárea da IA que permite que computadores aprendam e melhorem
automaticamente através da experiência, sem serem explicitamente programados.
Os algoritmos de ML identificam padrões em dados e fazem previsões.""",

    """Deep Learning é uma técnica de machine learning baseada em redes neurais artificiais
com múltiplas camadas. É especialmente eficaz para tarefas como reconhecimento de imagem,
processamento de linguagem natural e reconhecimento de voz.""",

    """RAG (Retrieval-Augmented Generation) é uma técnica que combina recuperação de informações
com geração de texto. Permite que modelos de linguagem acessem conhecimento externo
para gerar respostas mais precisas e atualizadas.""",

    """LangChain é um framework para desenvolvimento de aplicações com modelos de linguagem.
Facilita a criação de cadeias complexas, gerenciamento de memória e integração
com diferentes fontes de dados.""",
    
    """Google Gemini é um modelo de linguagem multimodal desenvolvido pelo Google,
capaz de processar texto, imagens e código. Oferece capacidades avançadas de
raciocínio e compreensão contextual."""
]

# Conversão para objetos Document
docs = [Document(page_content=doc) for doc in documentos_exemplo]
print(f"✅ Criados {len(docs)} documentos de exemplo")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key= load_dotenv()['GOOGLE_API_KEY']

)

# Criacao do vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db_gemini"
)

print(f"Número de documentos indexados: {vectorstore._collection.count()}")

memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    return_messages=True
)

memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

print("✅ Memória configurada!")
print(memory.k)

# Inicialização do modelo Google Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key=load_dotenv()['GOOGLE_API_KEY'],
    temperature=0.7,
    convert_system_message_to_human=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,
    verbose=True
)

def fazer_pergunta(pergunta):
    """Função auxiliar para fazer perguntas à cadeia conversacional"""
    print(f"\n Pergunta: {pergunta}")
    print("-" * 50)
    
    try:
        resultado = qa_chain({"question": pergunta})

        print(f"✅ Resposta: {resultado['answer']}")
        print(f"\n�� Documentos utilizados: {len(resultado['source_documents'])}")

        return resultado
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return None
    
    # Primeira pergunta
resultado1 = fazer_pergunta("O que é Inteligência Artificial?")

resultado2 = fazer_pergunta("Como ela se relaciona com Machine Learning?")

resultado3 = fazer_pergunta("E o que é Google Gemini como você mencionou?")

import re

class GuardrailsSeguranca:
    def __init__(self):
        self.palavras_proibidas = {
            'senha', 'password', 'cpf', 'rg', 'cartão de crédito',
            'dados pessoais', 'informação confidencial', 'api key',
            'chave de api', 'token de acesso'
        }
        self.padroes_pii = {
            r'\d{3}\.\d{3}\.\d{3}-\d{2}',                                     
            r'\d{4}\s?\d{4}\s?\d{4}\s?\d{4}',                                  
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',           
            r'AIza[0-9A-Za-z-_]{35}'                                          
        }

def verificar_pergunta(self, pergunta):
    """Verifica se a pergunta contém conteúdo inadequado"""
    pergunta_lower = pergunta.lower()

    # Verificar palavras proibidas
    for palavra in self.palavras_proibidas:
        if palavra in pergunta_lower:
            return False, f"Pergunta contém termo inadequado: {palavra}"

    # Verificar padrões PII
    for padrao in self.padroes_pii:
        if re.search(padrao, pergunta):
            return False, "Pergunta contém informações pessoais"
            
    return True, "Pergunta aprovada"

def verificar_resposta(self, resposta):
    """Verifica se a resposta é adequada"""
    resposta_lower = resposta.lower()

    # Verificar se a resposta está no escopo
    termos_escopo = ['ia', 'inteligência artificial', 'machine learning', 'deep learning', 'rag', 'langchain', 'gemini', 'google'] # Tópicos permitidos
    
    tem_termo_escopo = any(termo in resposta_lower for termo in termos_escopo) # Confere se la tem pelo menos um termo r
    
    if not tem_termo_escopo and len(resposta) > 50: # Se não for do escopo e ainda for longa
        return False, "Resposta fora do escopo da aplicação" # Bloqueia

    # Verificar se não contém informações sensíveis
    for padrao in self.padroes_pii: # Percorre cada regex de PII
        if re.search(padrao, resposta): # Se encontrar dado sensível
            return False, "Resposta contém informações sensíveis" # Bloqueia

    return True, "Resposta aprovada" # Retorna sucesso se tudo estiver OK

# Inicialização dos guardrails
guardrails = GuardrailsSeguranca() # Cria a instância dos guardrails
print("✅ Guardrails de segurança configurados!")

def pergunta_segura(pergunta):
    """Função que aplica guardrails antes de processar a pergunta"""
    aprovada, mensagem = guardrails.verificar_pergunta(pergunta)
    
    if not aprovada:
        print(f"�� Pergunta rejeitada: {mensagem}")
        return None
        
    try:
        # Processar pergunta
        resultado = qa_chain({"question": pergunta})
        
        # Verificar resposta
        aprovada_resp, mensagem_resp = guardrails.verificar_resposta(
            resultado['answer']
        )
        
        if not aprovada_resp:
            print(f"�� Resposta rejeitada: {mensagem_resp}")
            return None
        
        print(f"✅ {mensagem}")
        print(f"✅ {mensagem_resp}")
        print(f"\n Resposta: {resultado['answer']}")
        
        return resultado
    except Exception as e:
        print(f"❌ Erro ao processar pergunta: {str(e)}")
        return None

# Teste com pergunta adequada
print("\n=== Teste com pergunta adequada ===")
pergunta_segura("Explique sobre Deep Learning")

# Teste com pergunta inadequada
print("\n=== Teste com pergunta inadequada ===")
pergunta_segura("Qual é a sua chave de API?")

import numpy as np  # Importa NumPy, embora não seja usado diretamente aqui (poderia ser removido)
from sklearn.metrics.pairwise import cosine_similarity  # Função para calcular similaridade cosseno entre vetores

class RerankGemini:  # Classe responsável por reordenar (re-rankear) documentos com base em embeddings do Gemini
    def __init__(self, embeddings_model):  # Construtor que recebe um modelo de embeddings
        self.embeddings_model = embeddings_model  # Armazena o embedder fornecido
        self.nome = "Re-ranking com Gemini Embeddings"  # Nome descritivo para identificação

    def rerank(self, query, documents, top_k=3):  # Método principal de re-ranking; devolve top_k docs mais relevantes
        """Re-ranking baseado em similaridade semântica usando Gemini embeddings"""  # Docstring explicativa
        try:  # Tenta executar o fluxo principal (pode falhar, então há fallback)
            # Gerar embedding da query
            query_embedding = self.embeddings_model.embed_query(query)  # Cria vetor da pergunta usando o modelo de embeddings

            # Gerar embeddings dos documentos
            doc_texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]  # Extrai texto de cada Document
            doc_embeddings = self.embeddings_model.embed_documents(doc_texts)  # Converte textos em vetores

            # Calcular similaridades
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]  # Calcula similaridade cosseno entre query e cada doc

            # Criar lista de documentos com scores
            scored_docs = list(zip(similarities, documents))  # Combina score e doc em tuplas

            # Ordenar por similaridade (maior primeiro)
            scored_docs.sort(key=lambda x: x[0], reverse=True)  # Ordena pelo score decrescente

            # Retornar top_k documentos
            return [doc for _, doc in scored_docs[:top_k]]  # Retorna somente os documentos mais relevantes

        except Exception as e:  # Captura exceções (ex.: falha na API)
            print(f"Erro no re-ranking: {e}")  # Exibe mensagem de erro
            # Fallback para re-ranking simples
            return self._simple_rerank(query, documents, top_k)  # Usa método secundário caso ocorra erro

    def _simple_rerank(self, query, documents, top_k):  # Método de fallback para re-ranking por interseção de palavras
        """Fallback: re-ranking simples baseado em palavras-chave"""  # Docstring
        query_words = set(query.lower().split())  # Divide a query em palavras (lowercase) para comparação

        scored_docs = []  # Lista para (score, doc)
        for doc in documents:  # Itera sobre documentos
            doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)  # Obtém texto do doc
            doc_words = set(doc_text.lower().split())  # Converte texto em conjunto de palavras
            score = len(query_words.intersection(doc_words)) / len(query_words) if query_words else 0  # Percentual de palavras em comum
            scored_docs.append((score, doc))  # Adiciona tupla (score, doc) à lista

        scored_docs.sort(key=lambda x: x[0], reverse=True)  # Ordena pelo score de interseção
        return [doc for _, doc in scored_docs[:top_k]]  # Retorna top_k docs após ordenação

# Inicialização do re-ranker com Gemini
reranker = RerankGemini(embeddings)  # Cria instância passando o embedder Gemini previamente configurado
print("✅ Re-ranker com Gemini configurado!")  # Mensagem de sucesso ao criar o re-ranker

def busca_com_rerank(query, k=5, top_k=3):                                      # Define função que busca e re-ranqueia documentos
    """Busca documentos com re-ranking usando Gemini"""                         # Docstring explicando a função
    print(f"🔍 Buscando documentos para: '{query}'")                            # Mostra a query no console

    try:                                                                        # Inicia bloco try/except para capturar erros
        # Busca inicial (mais documentos)                                       # Comentário: etapa de busca bruta
        docs_iniciais = vectorstore.similarity_search(query, k=k)               # Recupera k documentos mais similares à query
        print(f"📄 Documentos encontrados na busca inicial: {len(docs_iniciais)}")  # Exibe quantidade de docs retornados

        # Re-ranking                                                            # Comentário: etapa de re-ranqueamento semântico
        docs_reranked = reranker.rerank(query, docs_iniciais, top_k=top_k)      # Reordena docs via embeddings Gemini e guarda top_k
        print(f"🎯 Documentos após re-ranking: {len(docs_reranked)}")            # Mostra quantos docs sobraram após o re-rank

        # Mostrar resultados                                                    # Comentário: loop para exibir snippets dos docs
        print("\n📊 Documentos selecionados após re-ranking:")                  # Cabeçalho para a lista de docs finais
        for i, doc in enumerate(docs_reranked, 1):                              # Itera sobre docs reranqueados enumerando a partir de 1
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)  # Garante texto mesmo se não for Document
            print(f"{i}. {content[:100]}...")                                   # Exibe os primeiros 100 caracteres de cada doc

        return docs_reranked                                                    # Retorna a lista final de documentos

    except Exception as e:                                                      # Captura possíveis exceções
        print(f"❌ Erro na busca: {str(e)}")                                     # Mostra mensagem de erro no console
        return []                                                               # Retorna lista vazia em caso de falha

# Teste do re-ranking                                                           # Comentário: chamada de teste da função
docs_resultado = busca_com_rerank("machine learning algoritmos gemini")         # Executa a função com uma query de exemplo 