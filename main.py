import os
import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

# 1. Cargar las variables de entorno desde el archivo .env
load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# 2. Inicializar el cliente de la API
client = InferenceClient(token=hf_token)

# 3. Inicializar la base de datos vectorial persistente local
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 4. Crear (o cargar si ya existe) la colección en el disco duro
collection = chroma_client.get_or_create_collection(name="libro_ia_collection")

# 5. El texto del libro
libro_texto = """
Capítulo 1: El Despertar de la IA. Durante años, la humanidad creyó que las máquinas
solo seguirían reglas lógicas estrictas. Sin embargo, la llegada de las redes neuronales
profundas cambió el paradigma. Los modelos ya no solo calculaban; ahora parecían intuir.
Esto generó un debate ético sin precedentes en la comunidad científica global.
"""

# 6. Dividimos el texto en Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    length_function=len
)
chunks = text_splitter.create_documents([libro_texto])

print(f"Generando embeddings para {len(chunks)} chunks y guardando en ChromaDB...\n")

# Listas temporales para hacer una inserción masiva eficiente
documents_texts = []
embeddings_list = []
documents_ids = []

# 7. Iteramos por cada chunk para obtener su embedding
for i, chunk in enumerate(chunks):
    embedding = client.feature_extraction(
        text=chunk.page_content,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Agrupamos los datos en nuestras listas
    documents_texts.append(chunk.page_content)
    embeddings_list.append(embedding)
    documents_ids.append(f"id_chunk_{i+1}")

# 8. Guardamos físicamente los vectores y los textos en ChromaDB
collection.add(
    embeddings=embeddings_list,
    documents=documents_texts,
    ids=documents_ids
)

print(f"¡Éxito! Se han guardado {collection.count()} elementos en la base de datos vectorial.")