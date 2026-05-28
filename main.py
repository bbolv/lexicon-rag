from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Simulamos un fragmento de tu libro
libro_texto = """
Capítulo 1: El Despertar de la IA. Durante años, la humanidad creyó que las máquinas
solo seguirían reglas lógicas estrictas. Sin embargo, la llegada de las redes neuronales
profundas cambió el paradigma. Los modelos ya no solo calculaban; ahora parecían intuir.
Esto generó un debate ético sin precedentes en la comunidad científica global.
"""

# 2. Configuramos el divisor
# chunk_size: caracteres máximos por trozo
# chunk_overlap: caracteres que se repiten entre trozos contiguos
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    length_function=len
)

# 3. Dividimos el texto
chunks = text_splitter.create_documents([libro_texto])

# 4. Mostramos el resultado
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content)