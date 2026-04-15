from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# 1. Load PDF
loader = PyPDFLoader("Prakash_Ranjan.pdf")
documents = loader.load()

# 2. Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=1000)
docs = splitter.split_documents(documents)

# 3. Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Vector DB
db = Chroma.from_documents(docs, embeddings)

# 5. Retriever
retriever = db.as_retriever(search_kwargs={"k": 8})

# 6. LLM
llm = Ollama(model="gemma:2b")

# 7. Manual RAG loop
while True:
    query = input("Ask: ")
    if query.lower() == "exit":
        break

    # ✅ FIXED LINE
    docs = retriever.invoke(query)

    # Build context
    context = "\n".join([doc.page_content for doc in docs])
    
    print("\n===== CONTEXT START =====\n")
    print(context)
    print("\n===== CONTEXT END =====\n")

    # Better prompt (reduces hallucination)
    prompt = f"""
You are a helpful assistant.
Answer ONLY from the context below.
If the answer is not present, say "Not found in document".

Context:
{context}

Question:
{query}
"""

    # Generate response
    response = llm.invoke(prompt)

    print("\nAnswer:", response, "\n")