import os
import ollama
from unstructured.partition.auto import partition
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
nltk.download('averaged_perceptron_tagger')

# configuration
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
DATA_FOLDER = 'data/' 
VECTOR_DB = []


# loading the files and chunking
def load_documents(folder_path):
    dataset = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            elements = partition(filename=filepath)
            text = "\n".join([el.text for el in elements if el.text])
            dataset.append({"filename": filename, "text": text})
            print(f"Loaded {filename} ({len(text.split())} words)")
        except Exception as e:
            print(f"Failed to parse {filename}: {e}")
    return dataset

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# -----------------------------

# embedding and storing chunks
def add_chunk_to_database(chunk):
    response = ollama.embed(model=EMBEDDING_MODEL, input=chunk)
    embedding = response['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

def embed_all_chunks(docs):
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            add_chunk_to_database(chunk)
        print(f"Embedded {len(chunks)} chunks from {doc['filename']}")

# -----------------------------

# retreival and similarity
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x**2 for x in a) ** 0.5
    norm_b = sum(x**2 for x in b) ** 0.5
    return dot / (norm_a * norm_b)

def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    scored = [(chunk, cosine_similarity(query_embedding, emb)) for chunk, emb in VECTOR_DB]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

# -----------------------------

# chat function
def chat():
    print("\nAsk me anything based on your consulting documents.\n")
    while True:
        query = input("Your question (or type 'exit'): ")
        if query.strip().lower() in ['exit', 'quit']:
            break

        results = retrieve(query)

        print("\nTop retrieved chunks:")
        for i, (chunk, score) in enumerate(results):
            print(f"  {i+1}. (score: {score:.2f}) {chunk[:100]}...")

        context = "\n".join([f"- {chunk}" for chunk, _ in results])

        prompt = f"""You are a helpful assistant.
Only use the context below to answer the question. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}
Answer:"""

        print("\nChatbot response:")
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            stream=True
        )
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        print("\n" + "-"*50)


# -----------------------------


# main
if __name__ == "__main__":
    print("Loading documents from:", DATA_FOLDER)
    documents = load_documents(DATA_FOLDER)

    print("\nEmbedding documents...")
    embed_all_chunks(documents)

    chat()
