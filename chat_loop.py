from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# LLM: Deepseek-coder
llm = Ollama(model="deepseek-coder")
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Cargar base vectorial
db = Chroma(persist_directory=".chroma", embedding_function=embedding)

# Crear cadena de QA
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

print("🧠 AnticoagulApp está listo. Escribe tu pregunta (o 'salir' para terminar):\n")

while True:
    pregunta = input("🔎 Tu pregunta: ")
    if pregunta.lower() in ["salir", "exit", "quit"]:
        print("👋 Hasta luego.")
        break
    respuesta = qa.invoke(pregunta)
    print("💬 Respuesta:", respuesta["result"], "\n")