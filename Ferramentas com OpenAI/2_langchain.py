from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os   
from pydantic import Field, BaseModel

load_dotenv()
api_key = os.getenv("OpenAi_API_Key")

class Destino(BaseModel):
    cidade: str = Field("Nome da cidade sugerida: ")
    motivo: str = Field("Motivo da sugestão: ")

parseador = JsonOutputParser(pydantic_object=Destino)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida} 
""",
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador.get_format_instructions()}
)

modelo = ChatOpenAI(
    model="gpt-5-2025-08-07",
    temperature=0.5,
    api_key=api_key
)

cadeia = prompt_cidade | modelo | parseador

resposta = cadeia.invoke(
    {
        "interesse": "praias e cultura"
    }
)
print(resposta)


