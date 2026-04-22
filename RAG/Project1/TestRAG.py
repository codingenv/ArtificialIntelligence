from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# =========================
# 1. LOAD PDF
# =========================
loader = PyPDFLoader("Prakash_Ranjan.pdf")
documents = loader.load()

print("\n✅ PDF Loaded Successfully")
print(f"Total Pages: {len(documents)}")

# =========================
# 2. SPLIT TEXT (BETTER CHUNKING)
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120
)

chunks = splitter.split_documents(documents)

print(f"\n✅ Total Chunks Created: {len(chunks)}")

# Optional: Debug first few chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Sample Chunk {i+1} ---")
    print(chunk.page_content[:300])

# =========================
# 3. EMBEDDINGS
# =========================
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# =========================
# 4. VECTOR DATABASE
# =========================
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

# =========================
# 5. RETRIEVER (IMPROVED)
# =========================
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6}
)

# =========================
# 6. LLM (LOCAL MODEL)
# =========================
llm = OllamaLLM(model="tinyllama")

# =========================
# 7. QUERY ENHANCEMENT
# =========================
def enhance_query(user_query: str) -> str:
    return f"""
    You are searching a resume.

    Find the most relevant details to answer:
    {user_query}

    Focus on:
    - current company
    - recent experience
    - profile summary
    """

# =========================
# 8. SIMPLE RE-RANKING
# =========================
def rerank_documents(docs):
    return sorted(
        docs,
        key=lambda d: (
            "202" in d.page_content,   # boost recent years
            "present" in d.page_content.lower(),
            "current" in d.page_content.lower()
        ),
        reverse=True
    )

# =========================
# 9. BUILD CONTEXT
# =========================
def build_context(docs, max_chars=2000):
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    ])
    return context[:max_chars]

# =========================
# 10. PROMPT TEMPLATE
# =========================
def create_prompt(context, query):
    return f"""
You are analyzing a resume.

Answer ONLY using the context below.

Rules:
- If current company is mentioned, return it clearly
- If not explicitly mentioned, infer from most recent experience
- If still not found, say "Not found in document"
- Keep answer short and precise

Context:
{context}

Question:
{query}
"""

# =========================
# 11. INTERACTIVE LOOP
# =========================
print("\n💬 Ask questions about the document (type 'exit' to quit)\n")

while True:
    user_query = input("Ask: ")

    if user_query.lower() == "exit":
        print("👋 Exiting...")
        break

    # Enhance query
    query = enhance_query(user_query)

    # Retrieve documents
    retrieved_docs = retriever.invoke(query)

    # Re-rank documents
    ranked_docs = rerank_documents(retrieved_docs)

    # Build context
    context = build_context(ranked_docs)

    # Debug (optional)
    print("\n===== CONTEXT PREVIEW =====")
    print(context[:500])

    # Create prompt
    prompt = create_prompt(context, user_query)

    # Get response
    response = llm.invoke(prompt)

    print("\n✅ Answer:", response, "\n")