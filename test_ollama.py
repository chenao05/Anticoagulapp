from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3")
respuesta = llm.invoke("¿Cuál es la dosis inicial de warfarina en fibrilación auricular?")
print("🧠 Respuesta de Ollama:")
print(respuesta)