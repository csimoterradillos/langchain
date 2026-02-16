import streamlit as st
from langchain_openai import ChatOpenAI

# Formato del curso:
# from langchain.schema import AIMessage, HumanMessage, SystemMessage
# from langchain.prompts import PromptTemplate

# Formato nuevo
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

# Mi variación sobre el programa del curso
# Yo uso load_dotenv para cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv
load_dotenv("/home/vant/cursos_udemy/langchain/apikeysrc.env")


plantilla = PromptTemplate(

    input_variables=['mensaje','historial'],
    template = """Eres un asistente amigable llamado Mi_Inteligente_ChatBot. 
    
    Historial de la conversación:
    
    {historial}
    
    Responde con versos alejandrinos a la siguiente pregunta:
    
    {mensaje}"""
)

# Configuración inicial
# Tema 23: Añadimos una sidebar para fijar temperatura de LLM
st.set_page_config(page_title="Chatbot Básico", 
                   page_icon="🤖",
                   layout="wide",
                   initial_sidebar_state="expanded")
st.title("🤖 Chatbot Básico con LangChain")
st.markdown("Este es un *chatbot de ejemplo* construido con LangChain + Streamlit. ¡Escribe tu mensaje abajo para comenzar!")

# Recrear el modelo con nuevos parámetros
# Lo trasladamos en tema 23 para poder ajustar dinámicamente los parámetros
# En el inicio de session_state y por cada iteración
# chat_model = ChatOpenAI(model="gpt-4o-mini", temperature="0.7")

# Inicializar el historial de mensajes en session_state
# Inicializar la memoria para poder seguir la conversación
# Los mensajes que se obtienen de LLM sobre el diccionario st.session_state
if "mensajes" not in st.session_state:
    # Todavía no hemos interactuado con el modelo
    # Inicializo session_state. Creo clave sobre el diccionario session_state
    # El diccionario es una lista. Cada mensaje un elemento de la lista
    st.session_state.mensajes = []

# Nuevo tarea 23
# Inicializar un parámetro para fijar LLM
if "llm" not in st.session_state:
    st.session_state.llm = "gpt-4o-mini"
# Tema 23: Por cada iteración ajustamos la temperatura al modelo
# Lo que implica redefinir la configuración del modelo a LangChain
with st.sidebar:
    st.header("Configuración")
    current_temperature = st.slider("Temperatura", 0.0, 1.0, 0.75, 0.1, key="temperature")
    # Tema 23: También podemos seleccionar el LLM
    # Por defecto gpt-4o-mini
    current_llm = st.selectbox("Selecciona el LLM", ["gpt-4o-mini", "gpt-4"], index=0, key="llm")
    # Tema 23: Ajustamos LLM y temperatura del LLM a invocar
    current_chat_model = ChatOpenAI(model=current_llm, temperature=current_temperature)
    # Tema 23: Introducimos la plantilla a usar
    current_chain = plantilla | current_chat_model

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
    # Tema 23: Añadimos la plantilla que hemos definido al inicio
    # st.session_state.mensajes.append(HumanMessage(content=pregunta))
    current_message = HumanMessage(content=pregunta)
    # Tema 23: Nos aseguramos que la pregunta actual se vaya al historico
    st.session_state.mensajes.append(current_message)
   
    historial_str = ""
    for msg in st.session_state.mensajes[:-1]: # Todos menos el último (que es la pregunta actual)
        role_label = "Assistant" if isinstance(msg, AIMessage) else "User"
        # Ignoramos system messages si los hubiera en la lista
        if not isinstance(msg, SystemMessage):
            historial_str += f"{role_label}: {msg.content}\n"

    # Generar respuesta usando el modelo LLM escogido
    # Le proporcionamos al modelo el historial de mensajes.
    # Se puede hacer mejor con plantillas de prompts
    # Formato antiguo:_
    # respuesta = chat_model.invoke(messages=st.session_state.mensajes)
    # Tema 23: Añadir la plantilla que hemos definido
    # respuesta = current_chat_model.invoke(st.session_state.mensajes)
    respuesta = current_chain.invoke({'mensaje': pregunta, 
                                      'historial': historial_str})

    # Mostrar la respuesta por pantalla
    with st.chat_message("assistant"):
        # Mostramos en formato markdown md la respuesta.
        st.markdown(respuesta.content)

    # Para mantener el contexto tenemos que añadir la respuesta al diccionario session_state
    # Son del tipo AIMessage (clase de LangChain) porque es la respuesta que nos devuelve LLM
    st.session_state.mensajes.append(AIMessage(content=respuesta.content))