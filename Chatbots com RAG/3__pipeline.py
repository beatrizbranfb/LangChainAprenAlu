import os 
import duckdb 
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores.utils import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()
con = duckdb.connect(database=':memory:', read_only=False)
con.execute("""
CREATE TABLE produtos (
    id INTEGER,
    nome VARCHAR,
    categoria VARCHAR,
    preco FLOAT,
    estoque INTEGER,
    descricao VARCHAR
);
""")

produtos_df = pd.DataFrame({
    'id': [101, 102, 103, 104],
    'nome': ['Laptop Gamer Z', 'Mouse Óptico Fast', 'Teclado Mecânico Pro', 'Monitor Curvo 34"'],
    'categoria': ['Eletrônicos', 'Acessórios', 'Acessórios', 'Eletrônicos'],
    'preco': [9500.00, 250.00, 800.00, 3200.00],
    'estoque': [15, 120, 60, 25],
    'descricao': [
        'Laptop de alta performance com placa de vídeo dedicada e 32GB RAM.',
        'Mouse com 16.000 DPI e design ergonômico para longas sessões.',
        'Teclado com switches mecânicos, RGB e layout ABNT2.',
        'Monitor ultrawide com alta taxa de atualização e cores vibrantes.'
    ]
})

print("Tabela 'produtos' criada e populada no DuckDB.")
print(con.execute("SELECT * FROM produtos;").fetchdf())

con.register('produtos_df', produtos_df)
con.execute('INSERT INTO produtos SELECT * FROM produtos_df')
df_produtos = con.execute("SELECT * FROM produtos;").fetchdf()
docs_sql = []
fonte_de_dados = "banco_de_dados_produtos_duckdb"
for _, row in df_produtos.iterrows():
    # Criar um texto descritivo a partir da linha
    page_content = f"Produto: {row['nome']}, Categoria: {row['categoria']}, Preço: R${row['preco']:.2f}. Em estoque: {row['estoque']} unidades. Descricao: {row['descricao']}"
    # Criar metadados estratégicos
    metadata = {
        'source': 'tabela_produtos_duckdb',
        'produto_id': row['id'],
        'categoria': row['categoria'],
        'preco': row['preco'],
        'ingestion_date': datetime.now().strftime('%Y-%m-%d')
    }
    docs_sql.append(
        Document(page_content=page_content, metadata=metadata)
    )
con.close()
print(f"Total de documentos gerados a partir do SQL: {len(docs_sql)}\n")
print("Exemplo de documento gerado a partir de uma linha do banco de dados:")
print(docs_sql[0])


from langchain_community.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader("relatorio_vendas.pdf", mode="elements")
docs_unstructured = loader.load()

print(f"Total de elementos extraídos: {len(docs_unstructured)}\n")

for doc in docs_unstructured:
    print(f"--- TIPO DE ELEMENTO: {doc.metadata.get('category')} ---")
    print(doc.page_content)
    print("\n")

from langchain.schema.document import Document

docs_com_metados = []

for doc in docs_unstructured:
    novos_metadados = doc.metadata.copy()
    novos_metadados['source'] = 'relatorio_vendas.pdf'
    novos_metadados['ingestion_date'] = datetime.now().strftime('%Y-%m-%d')
    novos_metadados['data_owner'] = 'Departamento de Vendas'
    
    doc_com_metadados = Document(
        page_content=doc.page_content,
        metadata=novos_metadados
    )
    docs_com_metados.append(doc_com_metadados)

print(f"Total de documentos com metadados: {len(docs_com_metados)}\n")
print(docs_com_metados[-1])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = text_splitter.split_documents(docs_com_metados)

print(f"Numero de documentos original: {len(docs_com_metados)}")
print(f"Numero de chunks gerados: {len(chunks)}")
print(chunks[2])

documentos_unidos = chunks + docs_sql
print(f"Total de documentos a serem indexados: {len(documentos_unidos)}")

documentos_filtrados = filter_complex_metadata(documentos_unidos)
print(f"Total de documentos filtrados: {len(documentos_filtrados)}")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma.from_documents(
    documents=documentos_filtrados,
    embedding=embeddings
)

pergunta_pdf = "Qual foi a receita com laptops?"
resultados_pdf = vector_store.similarity_search(pergunta_pdf, k=2)

print(f"Pergunta: {pergunta_pdf}\n")
for doc in resultados_pdf:
    print(f"Similaridade: {doc.page_content}")
    print(f"Fonte: {doc.metadata.get('source')}, Categoria: {doc.metadata.get('Category')}")

print("-"*20)
pergunta_sql = "Me fale sobre o teclado mecânico"
resultado_sql = vector_store.similarity_search(pergunta_sql, k=2)

print(f"Pergunta: {pergunta_sql}\n")
for doc in resultado_sql:
    print(f"Similaridade: {doc.page_content}")
    print(f"Fonte: {doc.metadata.get('source')}, Categoria: {doc.metadata.get('categoria')}")