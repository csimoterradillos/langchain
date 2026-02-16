from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

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
¿Que queremos hacer con el LLM?
En este caso es un chatbot con una pregunta a realizar
'''
pregunta = "¿En que año el ser humano llegó a la Luna?"

'''
Usamos el model de LLM de openAI
'''
print("Pregunta_openAI:::", pregunta)

respuesta_langchain = llm_chatOpenAI.invoke(pregunta)
print("Respuesta_openAI:::", respuesta_langchain.content)

'''
Usamos el model de LLM de Google
'''
print("Pregunta_Google:::", pregunta)

respuesta_google = llm_chatGoogle.invoke(pregunta)
print("Respuesta_Google:::", respuesta_google.content)
