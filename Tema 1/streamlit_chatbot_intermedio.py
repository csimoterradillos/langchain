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

# Inicializar el historial de mensajes en session_state
# Inicializar la memoria para poder seguir la conversación
# Los mensajes que se obtienen de LLM sobre el diccionario st.session_state
if "mensajes" not in st.session_state:
    # Todavía no hemos interactuado con el modelo
    # Inicializo session_state. Creo clave sobre el diccionario session_state
    # El diccionario es una lista. Cada mensaje un elemento de la lista
    st.session_state.mensajes = []


for msg in st.session_state.mensajes:
    # Volvemos a leer toda la lista contenida en diccionario session_state en mensajes
    # Primero, que tipo de mensaje es: AIMessage o HumanMessage o SystemMessage
    if isinstance(msg, SystemMessage):
        # Mensaje del sistema
        continue

    role = "assistant" if isinstance(msg, AIMessage) else "user"

    # Mostrar el mensaje por pantalla dependiendo del role
    with st.chat_message(role):
        # Mensaje en formato markdown md
        st.markdown(msg.content)

# Recoger el mensaje de usuario en un cuadro de texto
# Hace sleep del script hasta que el usuario introduce info
pregunta = st.chat_input("Escribe tu mensaje aquí...")

# Que vamos a hacer con la pregunta
if pregunta:
    # El usuario ha introducido una pregunta
    # Añadir el mensaje del usuario al historial
    with st.chat_message("user"):
        st.markdown(pregunta)
        
    # Almacenamos el mensaje en la memoria de Streamlit
    # Tenemos que indicarle que tipo de mensaje añadimos. En este caso
    # una pregunta de un usuario por tanto HumanMessage (clase de LangChain)
    st.session_state.mensajes.append(HumanMessage(content=pregunta))


    # Generar respuesta usando el modelo LLM escogido
    # Le proporcionamos al modelo el historial de mensajes.
    # Se puede hacer mejor con plantillas de prompts
    # Formato antiguo:_
    # respuesta = chat_model.invoke(messages=st.session_state.mensajes)
    respuesta = chat_model.invoke(st.session_state.mensajes)

    # Mostrar la respuesta por pantalla
    with st.chat_message("assistant"):
        # Mostramos en formato markdown md la respuesta.
        st.markdown(respuesta.content)

    # Para mantener el contexto tenemos que añadir la respuesta al diccionario session_state
    # Son del tipo AIMessage (clase de LangChain) porque es la respuesta que nos devuelve LLM
    st.session_state.mensajes.append(AIMessage(content=respuesta.content))