import os
import chromadb
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# 1. Configuración inicial y carga de credenciales
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
client = InferenceClient(token=hf_token)

# 2. Conectarse a la base de datos persistente que ya creamos en main.py
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="libro_ia_collection")

# 3. La pregunta del usuario (Query)
# Pregunta usando un sinónimo conceptual
pregunta_usuario = "¿Qué provocó la evolución en el pensamiento de los sistemas?"

print(f"Pregunta: '{pregunta_usuario}'")
print("Transformando la pregunta en un vector de búsqueda...")

# 4. Convertimos la pregunta en su embedding correspondiente
query_embedding = client.feature_extraction(
    text=pregunta_usuario,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

print("Buscando en el espacio de fase de ChromaDB...\n")

# 5. Consultamos a la base de datos por los 2 fragmentos más cercanos (n_results=2)
resultados = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

# 6. Unimos los fragmentos recuperados en una sola cadena de texto (el Contexto)
fragmentos = resultados['documents'][0]
contexto_extraido = "\n".join(fragmentos)

print("2. Contexto relevante recuperado de ChromaDB.")

# 6. Generación (Generation) - Estructura moderna de Chat Messages
print("3. Enviando pregunta + contexto al LLM generativo en Hugging Face...\n")

# Construimos las instrucciones de restricción para el sistema
instrucciones_sistema = f"""Eres un asistente experto en Inteligencia Artificial y Ciencia. 
Tu único objetivo es responder la pregunta del usuario utilizando exclusivamente el contexto proporcionado abajo.

[CONTEXTO]
{contexto_extraido}

Instrucciones estrictas: Responde con claridad, fluidez y basándote únicamente en el contexto anterior. Si el contexto no contiene la información para responder, di textualmente: 'No encontré esa información en el documento'."""

# 7. Usamos la API unificada de Chat Completions
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