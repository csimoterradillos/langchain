from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

'''
En el curso con formato antiguo
from langchain.prompts import PromptTemplate
'''
from langchain_core.prompts import PromptTemplate
'''
Chains
En el curso, se importa con formato antiguo
from langchain.chains import LLMChain
ahora se pueden usar dos enfoques:
1. Usar la librería langchain_classic
from langchain_classic.chains import LLMChain
2. Usar la nueva libreria langchain_openai y el operador pipeline: 
from langchain_openai import ChatOpenAI
plantilla | ChatOpenAI()
'''

'''
Mi variación sobre el programa del curso
Yo uso load_dotenv para cargar variables de entorno desde un archivo .env
'''
from dotenv import load_dotenv
load_dotenv("/home/vant/cursos_udemy/langchain/apikeysrc.env")

'''
Modelo de LLM a usar dentro de ChatOpenAI
Baratito por ser gpt-4o-mini
Bastante creativo por temperatura cercana a 1
'''
llm_chatOpenAI = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
llm_chatGoogle = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

'''
    la lista input_variables es un array de elementos que serán nuestras variables dinámicas
    template es la plantilla del prompt con la variable dinámica nombre.
    Lo de Asistente es decirle al prompt que estamos esperando una respuesta del LLM
'''
plantilla = PromptTemplate(

    input_variables=['nombre'],
    template='Saluda al usuario. \nNombre del usuario: {nombre}\nAsistente:'
)

'''
Chains
Formato antiguo en el curso:
llm_chain = LLMChain(llm=llm_chatOpenAI, prompt=plantilla)
Ahora LCEL: hacemos referencia al modelo LLM que queremos usar enlazado con pipeline
'''
llm_chain = plantilla | llm_chatOpenAI


'''
invocar cadena
en curso formato antiguo:
resultado = llm_chain.run(nombre="Carlos")
print(resultado)
En formato nuevo LCEL: se deben definir tantas tuplas como variables dinámicas usemos con su valor correspondiente
'''
resultado = llm_chain.invoke({'nombre': 'Carlos'})
print(resultado.content)