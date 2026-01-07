import os
import duckdb
import pandas as pd
from datetime import datetime
import keySecure
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

keySecure.get_google_api_key(self=None)

loader = UnstructuredPDFLoader("documentos/TCU_Atividades2022.pdf")

docs_unstructured = loader.load()

print(f"Número de páginas carregadas: {len(docs_unstructured)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)