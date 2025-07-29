from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

# Ruta de la carpeta donde están tus PDFs
folder_path = "./docs"

# Leer y dividir todos los documentos PDF
all_chunks = []

for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)

print(f"✅ Se cargaron {len(all_chunks)} fragmentos de texto.")

# Embeddings con nomic-embed-text
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Crear base de datos vectorial y guardar
db = Chroma.from_docs(docs=all_chunks, embedding=embedding, persist_directory="./chroma")
db.persist()
print("✅ Base de datos vectorial creada y guardada en ./chroma")