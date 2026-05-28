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

# 6. Desplegamos los fragmentos recuperados
# ChromaDB nos regresa una estructura con 'documents' y 'distances'
for i in range(len(resultados['documents'][0])):
    texto_recuperado = resultados['documents'][0][i]
    distancia = resultados['distances'][0][i]
    id_chunk = resultados['ids'][0][i]
    
    print(f"🎯 --- Fragmento Recuperado {i+1} ({id_chunk}) ---")
    print(f"Distancia geométrica (menor es más cercano): {distancia:.4f}")
    print(f"Contenido: {texto_recuperado}\n")