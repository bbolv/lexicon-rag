import os
import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

# Load the environment variables from the .env file
load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# Initialize the Hugging Face Interface client
client = InferenceClient(token=hf_token)

# Initialize the ChromaDB client (persistent local vector database)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create the collection on the hard disk (or load if it already exists)
collection = chroma_client.get_or_create_collection(name="libro_ia_collection")

# Load the text of the book
libro_texto = """
Capítulo 1: El Despertar de la IA. Durante años, la humanidad creyó que las máquinas
solo seguirían reglas lógicas estrictas. Sin embargo, la llegada de las redes neuronales
profundas cambió el paradigma. Los modelos ya no solo calculaban; ahora parecían intuir.
Esto generó un debate ético sin precedentes en la comunidad científica global.
"""

# Create a text splitter to divide the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    length_function=len
)
chunks = text_splitter.create_documents([libro_texto])

print(f"Generando embeddings para {len(chunks)} chunks y guardando en ChromaDB...\n")

# Temporal lists to make an efficient bulk insertion
documents_texts = []
embeddings_list = []
documents_ids = []

# Iterate over each chunk to get its embedding
for i, chunk in enumerate(chunks):
    embedding = client.feature_extraction(
        text=chunk.page_content,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Group the data in our lists
    documents_texts.append(chunk.page_content)
    embeddings_list.append(embedding)
    documents_ids.append(f"id_chunk_{i+1}")

# Save the embeddings and texts physically in ChromaDB
collection.add(
    embeddings=embeddings_list,
    documents=documents_texts,
    ids=documents_ids
)

print(f"¡Éxito! Se han guardado {collection.count()} elementos en la base de datos vectorial.")