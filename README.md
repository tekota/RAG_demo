# RAG_demo

⚠️ Since this runs locally, you'll need to install Ollama's models for it to work. If you can't be bothered to do so, I am linking a video demonstrating how the RAG system works:
https://drive.google.com/file/d/1bZ5BXw1cQg5MVs4R-WfwMd4QS45zHRSt/view?usp=sharing

Using local embedding and machine learning models from Ollama, this RAG model loads, parses, chunks and retrieves context from provided files in a selected data folder. 
Using cosine similarity to check the similiarity between the embedding vectors in the database and the prompt the user gave to the chatbot.

The script parses, chunks and embedds the files into a local vector database. After the user asks a question, it will search for relevant context using the vector database by looking for vectorial similarities and according to these, output a contextually relevant answer. And if it doesn't know, it will clearly says it doesn't know instead of outputting false or unsure information.

The libraries that are being used are:
- ollama
- langchain
- unstructred
- nltk
- poppler
