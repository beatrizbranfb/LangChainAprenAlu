# Aprendendo LangChain
- Os documentos postados servem apenas para aprendizado pessoal
- Colocar a chave em um arquivo .env

# Palavras chave
- RAG: Geração aumentada por recuperação, combina a recuperação de informações relevantes de fontes externas com a geração de respostas
- Embedding: transformam dados em vetores
- Fine-tuning: adaptar modelos pré treinados para algo específico desejado
- Pipeline de ingestão: etapas que permitem a movimentação e transformação de dados desde suas fontes até um repositório para análise
- ETL: Extract, Transform, Load
- CRC: Utilização de histórico de conversas para contextualizar, mantendo um fluxo de conversa
- Guardrails de segurança: mecanismos para gerar limites definidos, evitando respostas indesejadas

# Tipos de índices
- Flat: compara com todos os vetores, garantindo precisão perfeita, porém sendo muito lento
- IVF: agrupa os vetores em clusters, buscando apenas nos mais relevantes.
- HNSW: constrói um grafo multicamadas para busca mais rápida e eficiente.

