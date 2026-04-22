from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import json
import re

# =========================
# 1. LOAD PDF
# =========================
loader = PyPDFLoader("Prakash_Ranjan.pdf")
documents = loader.load()

# =========================
# 2. SPLIT TEXT (OPTIMIZED)
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)
chunks = splitter.split_documents(documents)

# =========================
# 3. EMBEDDINGS + VECTOR DB
# =========================
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4}
)

# =========================
# 4. LLM (LOW RAM SAFE)
# =========================
llm = OllamaLLM(
    model="phi",   # better than tinyllama
    temperature=0  # IMPORTANT → stable output
)

# =========================
# 5. BUILD CONTEXT
# =========================
def get_context(query):
    docs = retriever.invoke(query)

    context = "\n\n".join([d.page_content for d in docs])

    return context[:1200]  # reduced for low RAM


# =========================
# 6. SAFE JSON EXTRACTION
# =========================
def extract_json(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return None


# =========================
# 7. EXTRACT STRUCTURED DATA
# =========================
def extract_resume_data():
    context = get_context("Extract complete resume details")

    prompt = f"""
You are a JSON generator.

Return ONLY valid JSON. No explanation. No extra text.

STRICT FORMAT:
{{
  "name": "",
  "current_company": "",
  "total_experience": "",
  "skills": [],
  "recent_role": ""
}}

Rules:
- Do NOT add any text before or after JSON
- Do NOT explain anything
- Use exact keys only
- If not found, keep empty string

Resume:
{context}
"""

    # Retry mechanism (IMPORTANT)
    for _ in range(2):
        response = llm.invoke(prompt)
        data = extract_json(response)

        if data:
            return data

    print("⚠️ JSON parsing failed, raw output:")
    print(response)
    return {}


# =========================
# 8. SKILL MATCHING
# =========================
def calculate_match(candidate_skills, job_skills):
    candidate_skills = [s.lower() for s in candidate_skills]
    job_skills = [s.lower() for s in job_skills]

    matched = set(candidate_skills) & set(job_skills)

    if len(job_skills) == 0:
        return 0, []

    score = (len(matched) / len(job_skills)) * 100

    return round(score, 2), list(matched)


# =========================
# 9. Q&A SYSTEM
# =========================
def ask_question(query):
    context = get_context(query)

    prompt = f"""
Answer ONLY using the context below.

If not found, say "Not found in document".

Context:
{context}

Question:
{query}
"""

    return llm.invoke(prompt)


# =========================
# 10. MAIN MENU
# =========================
while True:
    print("\n====== AI Resume System ======")
    print("1. Extract Resume Data")
    print("2. Ask Question")
    print("3. Match with Job")
    print("4. Exit")

    choice = input("Enter choice: ")

    # -------------------------
    # OPTION 1: EXTRACT
    # -------------------------
    if choice == "1":
        data = extract_resume_data()
        print("\n📄 Extracted Data:")
        print(json.dumps(data, indent=2))

    # -------------------------
    # OPTION 2: Q&A
    # -------------------------
    elif choice == "2":
        q = input("Ask: ")
        answer = ask_question(q)
        print("\n💬 Answer:", answer)

    # -------------------------
    # OPTION 3: MATCHING
    # -------------------------
    elif choice == "3":
        data = extract_resume_data()

        if not data or "skills" not in data:
            print("⚠️ Could not extract skills")
            continue

        job_input = input("Enter job skills (comma separated): ")
        job_skills = [s.strip() for s in job_input.split(",")]

        score, matched = calculate_match(data["skills"], job_skills)

        print(f"\n🎯 Match Score: {score}%")
        print(f"✅ Matched Skills: {matched}")

    elif choice == "4":
        print("👋 Exiting...")
        break

    else:
        print("❌ Invalid choice")