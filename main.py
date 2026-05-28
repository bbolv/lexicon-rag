import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

# 1. Cargar las variables de entorno desde el archivo .env
load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# 2. Inicializar el cliente de la API
client = InferenceClient(token=hf_token)

# 3. El texto deL libro
libro_texto = """
Capítulo 1: El Despertar de la IA. Durante años, la humanidad creyó que las máquinas
solo seguirían reglas lógicas estrictas. Sin embargo, la llegada de las redes neuronales
profundas cambió el paradigma. Los modelos ya no solo calculaban; ahora parecían intuir.
Esto generó un debate ético sin precedentes en la comunidad científica global.
"""

# 4. Dividimos el texto en Chunks
# chunk_size: caracteres máximos por trozo
# chunk_overlap: caracteres que se repiten entre trozos contiguos
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    length_function=len
)
chunks = text_splitter.create_documents([libro_texto])

print(f"Enviando {len(chunks)} chunks a la API de Hugging Face...\n")

# 5. Iteramos por cada chunk para obtener su representación matemática (Embedding)
for i, chunk in enumerate(chunks):
    # Llamamos a la API gratuita para extraer las características vectoriales
    embedding = client.feature_extraction(
        text=chunk.page_content,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # El embedding devuelto es una lista de floats (coordenadas en el espacio de la IA)
    print(f"--- Chunk {i+1} transformado ---")
    print(f"Texto: {chunk.page_content[:40]}...")
    print(f"Tamaño del vector obtenido: {len(embedding)} dimensiones")
    print(f"Muestra del vector (primeros 3 números): {embedding[:3]}\n")