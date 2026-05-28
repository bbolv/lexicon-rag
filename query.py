import os
import chromadb
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load the environment variables from the .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
client = InferenceClient(token=hf_token)

# Initialize the ChromaDB client that has been created in main.py
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="libro_ia_collection")

# Get the user's question
# Question using a conceptual synonym
pregunta_usuario = "¿Qué provocó la evolución en el pensamiento de los sistemas?"

print(f"Pregunta: '{pregunta_usuario}'")
print("Transformando la pregunta en un vector de búsqueda...")

# Convert the question to its corresponding embedding
query_embedding = client.feature_extraction(
    text=pregunta_usuario,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

print("Buscando en el espacio de fase de ChromaDB...\n")

# Query the database for the 2 closests chunks
resultados = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

# Join the retrieved chunks into a single string (the Context)
fragmentos = resultados['documents'][0]
contexto_extraido = "\n".join(fragmentos)

print("2. Contexto relevante recuperado de ChromaDB.")

# Generation using modern Chat Messages structure
print("3. Enviando pregunta + contexto al LLM generativo en Hugging Face...\n")

# Build the restriction instructions for the system
instrucciones_sistema = f"""Eres un asistente experto en Inteligencia Artificial y Ciencia. 
Tu único objetivo es responder la pregunta del usuario utilizando exclusivamente el contexto proporcionado abajo.

[CONTEXTO]
{contexto_extraido}

Instrucciones estrictas: Responde con claridad, fluidez y basándote únicamente en el contexto anterior. Si el contexto no contiene la información para responder, di textualmente: 'No encontré esa información en el documento'."""

# Use the unified Chat Completions API to generate the response
respuesta_chat = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": instrucciones_sistema},
        {"role": "user", "content": pregunta_usuario}
    ],
    max_tokens=150,
    temperature=0.3
)

print("🎯 --- Respuesta Final del RAG ---")
print(respuesta_chat.choices[0].message.content.strip())