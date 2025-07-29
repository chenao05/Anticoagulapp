from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# LLM
llm = Ollama(model="deepseek-coder")
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Cargar base vectorial
db = Chroma(persist_directory=".chroma", embedding_function=embedding)

# Crear cadena de QA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Consulta directa de prueba
pregunta = "¿Cuál es la dosis inicial recomendada de warfarina en fibrilación auricular?"
respuesta = qa.run(pregunta)
print("💬 Respuesta:", respuesta)