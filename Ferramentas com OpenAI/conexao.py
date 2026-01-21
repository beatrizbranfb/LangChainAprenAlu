from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OpenAi_API_Key")

numero_dias = 7
numero_adultos = 2  
numero_criancas = 2
cidade_destino = "Salvador"
mes_viagem = "Julho"


prompt = f"Crie um roteiro de viagem de {numero_dias} dias, para uma família com {numero_adultos} adultos e {numero_criancas} crianças, visitando {cidade_destino} durante o mês de {mes_viagem}"

cliente = OpenAI(api_key=api_key)

resposta = cliente.chat.completions.create(
    model="gpt-5.2",
    messages=[
        {"role": "system", "content": "Você é um assistente de planejamento de viagens."},
        {"role": "user", "content": prompt}
    ]
)

resposta_texto = resposta.choices[0].message.content
print(resposta_texto)