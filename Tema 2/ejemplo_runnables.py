from langchain_core.runnables import RunnableLambda

# Funcion lambda que devuelve el literal de un número
# Con RunnableLambda este lambda es runnable y
# concatenarlo dentro de una chain de LangChain
paso1 = RunnableLambda(lambda x: f"Numero {x}")


def duplicar_text(texto):
    return [texto] * 2

# Convierto duplicar_text en otro runnable de LangChain, para poder concatenarlo con paso1
paso2 = RunnableLambda(duplicar_text)

# Aquí el pipeline es el operador de LangChain que hace que el resultado de paso1
# se convierta en la invocación de paso2.invoke(paso1), mas o menos.
cadena = paso1 | paso2

resultado = cadena.invoke(167)
print(resultado)