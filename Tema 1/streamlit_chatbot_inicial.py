import streamlit as st
from langchain_openai import ChatOpenAI

'''
Formato del curso:
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
'''

'''
Formato nuevo
'''
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

'''
Mi variación sobre el programa del curso
Yo uso load_dotenv para cargar variables de entorno desde un archivo .env
'''
from dotenv import load_dotenv
load_dotenv("/home/vant/cursos_udemy/langchain/apikeysrc.env")

# Configuración inicial
st.set_page_config(page_title="Chatbot Básico", page_icon="🤖")
st.title("🤖 Chatbot Básico con LangChain")
st.markdown("Este es un *chatbot de ejemplo* construido con LangChain + Streamlit. ¡Escribe tu mensaje abajo para comenzar!")

# Recrear el modelo con nuevos parámetros
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature="0.7")

