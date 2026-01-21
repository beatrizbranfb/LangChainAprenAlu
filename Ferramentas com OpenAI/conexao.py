from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("OpenAi_API_Key")

numero_dias = 7
numero_adultos = 2  
numero_criancas = 2
cidade_destino = "Salvador"
mes_viagem = "Julho"

modelo_prompt = PromptTemplate(
    template="""
Crie um roteiro de viagem de {numero_dias} dias, 
para uma família com {numero_adultos} adultos e {numero_criancas} crianças, 
visitando {cidade_destino},
 durante o mês de {mes_viagem}
 """
)
prompt = modelo_prompt.format(
    numero_dias=numero_dias,    
    numero_adultos=numero_adultos,
    numero_criancas=numero_criancas,
    cidade_destino=cidade_destino,
    mes_viagem=mes_viagem
)
modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)

resposta = modelo.invoke(prompt)
print(resposta.content)