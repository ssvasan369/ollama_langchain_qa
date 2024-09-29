import os
import sys

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, JSONLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
vectordb_path = os.getenv("VECTORDB_PATH", ".data")
ollama_obj = Ollama(base_url=ollama_url, model=ollama_model)
ollama_embeddings = OllamaEmbeddings(base_url=ollama_url, model=ollama_model)
docs_dir = os.getenv("DOCS_DIR", "./data")

print("Loading Ollama model type %s from %s" % (ollama_model, ollama_url))

# print vars
print(f"Using docs at {docs_dir}")
print(f"Using vectordb at {vectordb_path}")
print(f"Using Ollama at {ollama_url} with model {ollama_model}")

documents = []
for file in os.listdir(docs_dir):
    file_path = os.path.join(docs_dir, file)
    
    if file.endswith('.json'):
        loader = JSONLoader(file_path)
        documents.extend(loader.load())
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())


text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
chunked_documents = text_splitter.split_documents(documents)

print(f"Loaded {len(documents)} documents")
print(f"Split documents into {len(chunked_documents)} chunks")

print(f"Creating vectorstore")
vectordb = Chroma.from_documents(
    documents, 
    embedding=ollama_embeddings, 
    persist_directory=vectordb_path)
vectordb.persist()

print(f"Created vectorstore")

qachain=RetrievalQA.from_chain_type(ollama_obj, retriever=vectordb.as_retriever())

print(f"Created QA chain")


yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the OllamaBot! Ask me a question about the documents in the docs folder.')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        print('Query cannot be empty. Please enter a valid query.')
        continue
    result = qachain.invoke({"query": query})

    print(f"{white}Answer: " + result["result"])