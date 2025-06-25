# RAG_demo

Using local embedding and machine learning models from Ollama, this RAG model loads, parses, chunks and retrieves context from provided files in a selected data folder. 
Using cosine similarity to check the similiarity between the embedding vectors in the database and the prompt the user gave to the chatbot.

The script parses, chunks and embedds the files into a local vector database. After the user asks a question, it will search for relevant context using the vector database by looking for vectorial similarities and according to these, output a contextually relevant answer. And if it doesn't know, it will clearly says it doesn't know instead of outputting false or unsure information.

The libraries that are being used are:
- ollama
- langchain
- unstructred
- nltk
- poppler
