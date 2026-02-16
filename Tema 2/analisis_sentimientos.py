import json
import re
from langchain_core.runnables import RunnableLambda,RunnableParallel
from langchain_openai import ChatOpenAI

# Mi variación sobre el programa del curso
# Yo uso load_dotenv para cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv
load_dotenv("/home/vant/cursos_udemy/langchain/apikeysrc.env")

# Configuración del modelo
# temperatura a 0 para que sea más imaginativo al hablar de sentimientos
# parezca mas humano
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def preprocess_text(text):

    return text.strip()[:500]


# Convertimos preprocess_text en una función runnable de LangChain
preprocessor = RunnableLambda(preprocess_text)


def translate_question(text):
    """Intento de manejar varios idiomas en las consultas"""

    prompt = f"""Identifica el idioma del texto y traduce a castellano.
    Responde ÚNICAMENTE con un objeto JSON válido, sin bloques de código markdown.
    
    Formato requerido:
    {{"language": "nombre_del_idioma", "translation": "texto_traducido"}}

    Texto: {text}"""

    response = llm.invoke(prompt)
    
    # Limpia el contenido eliminando bloques markdown
    content = response.content.strip()
    content = re.sub(r'^```(?:json)?[\s\n]*|[\s\n]*```$', '', content, flags=re.MULTILINE)
    content = content.strip()
 
    print(f"DEBUG - cleaned content: '{content}'")
    
    # Parsea el JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"ERROR parseando JSON: {e}")
        print(f"Contenido recibido: {content}")
        # Retorna un valor por defecto
        return {"language": "desconocido", "translation": text}
    

translate_branch = RunnableLambda(translate_question)


def find_translated_language(data):
    return data["language"]


id_language_branch = RunnableLambda(find_translated_language)


def generate_summary(data):
    """Genera un resumen conciso del texto"""
    prompt = f"Resume en una sola oración: {data['translation']}"
    response = llm.invoke(prompt)
    return response.content


summary_branch = RunnableLambda(generate_summary)


def analyze_sentiment(data):
    """Analiza el sentimiento y devuelve resultado estructurado"""
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde ÚNICAMENTE en formato JSON válido:
    {{"sentimiento": "positivo|negativo|neutro", "razon": "justificación breve"}}
    
    Texto: {data['translation']}"""
    
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"sentimiento": "neutro", "razon": "Error en análisis"}


analyze_branch = RunnableLambda(analyze_sentiment)


def merge_results(data):
    """Combina los resultados de ambas ramas en un formato unificado"""
    return {
        "idioma_de_consulta": data["idioma_original_consulta"],
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon": data["sentimiento_data"]["razon"]
    }


merge_branch = RunnableLambda(merge_results)


'''
Versión original de la lección
Tenemos que incorporar la traducción previa a castellano
def process_one(t):
    resumen = generate_summary(t)              # Llamada 1 al LLM
    sentimiento_data = analyze_sentiment(t)    # Llamada 2 al LLM
    return merge_results({
        "resumen": resumen,
        "sentimiento_data": sentimiento_data
    })
'''

'''
Lección 28:
Al usar RunnableParallels ya no necesitaremos process_one
Porque procesaremos en paralelo tanto el resumen como el análisis de sentimientos
Llamaremos al traductor, los hilos en paralelo y finalmente el merge directamente en una pipe LangChain
def process_one(t):
    json_translation = translate_question(t)                                 # Llamada 0 traducción por el LLM 
    resumen = generate_summary(json_translation["translation"])              # Llamada 1 al LLM
    sentimiento_data = analyze_sentiment(json_translation["translation"])    # Llamada 2 al LLM
    return merge_results({
        "idioma_original_consulta": json_translation["language"],
        "resumen": resumen,
        "sentimiento_data": sentimiento_data
    })

# Convertir en Runnable
process = RunnableLambda(process_one)
'''

'''
Lección 28:
Invocación del RunnableParallels entre el resumen y el analisis
'''
main_branch = RunnableParallel({"idioma_original_consulta": id_language_branch, "resumen": summary_branch, "sentimiento_data": analyze_branch})


# La cadena completa
'''
Lección 28:
Reformateo la cadena incluyendo el lambda del traductor, el lambda del parallel y el del merge
chain = preprocessor | process
'''
pre_chain = preprocessor | translate_branch
chain = pre_chain | main_branch | merge_branch


# Prueba con diferentes textos
textos_prueba = [
    "¡Me encanta este producto! Funciona perfectamente y llegó muy rápido.",
    "El servicio al cliente fue terrible, nadie me ayudó con mi problema.",
    "El clima está nublado hoy, probablemente llueva más tarde.",
    "I demand better attention to this problem",
    "La qualité du produit m'a laissé indifférent. Je n'aurais pas dû l'acheter."
    ""
]

for texto in textos_prueba:
    resultado = chain.invoke(texto)
    print(f"Texto: {texto}")
    print(f"Resultado: {resultado}")
    print("-" * 50)
