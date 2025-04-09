import os
from langchain_google_genai import ChatGoogleGenerativeAI

from pdfminer.high_level import extract_text
import docx2txt
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from supabase import create_client, Client

from dotenv import dotenv_values

config = dotenv_values(".env")

# print("Config", config)

# Set up Google Gemini API key
os.environ["GOOGLE_API_KEY"] = config["API_KEY"]

# Initialize LangChain Gemini model
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)


def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)


def extract_text_from_docx(docx_path):
    return docx2txt.process(docx_path)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Supabase credentials
SUPABASE_URL = config["SUPABASE_URL"]
SUPABASE_KEY = config["SUPABASE_ANON_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# Create FAISS vector store
def create_vector_store(resume_texts):
    vectorstore = FAISS.from_texts(resume_texts, embedding_model)
    retriever = vectorstore.as_retriever()
    return retriever


def retrieve_top_resumes(job_desc, retriever, top_k=6):
    return retriever.get_relevant_documents(job_desc, k=top_k)


def rank_resumes_with_gemini(job_desc, top_resumes):
    prompt = f"""
    ### ROLE:
You are an advanced AI Resume Ranking system that helps recruiters find the best candidates for a given job description. You follow a **Retrieval-Augmented Generation (RAG) approach**, where resumes are retrieved from a knowledge base using semantic search before ranking them based on relevance.

### OBJECTIVE:
Rank the provided resumes in order of relevance to the given **job description** and explain why each resume was ranked in that position. Highlight the strengths and weaknesses of each candidate in relation to the job. Exclude any resumes that do not match the job description well.

---

---

### **INPUT DATA**:
#### **1️⃣ Job Description:**
{job_desc}

#### **2️⃣ Retrieved Candidate Resumes (Top k Matches):**
{[top_resume.page_content for top_resume in top_resumes]}

---

### **INSTRUCTIONS FOR RANKING**:
1. **Analyze the Job Description**:
   - Identify key requirements (skills, experience, education, certifications, etc.).
   - Note any specific technologies, tools, or domain expertise required.

2. **Compare Resumes Against Job Requirements**:
   - Evaluate how well each resume **matches the job requirements**.
   - Identify **missing or extra qualifications**.
   - Consider **years of experience**, **relevant projects**, **technical skills**, and **soft skills**.

3. **Rank the Candidates**:
   - Sort resumes from **most to least relevant** based on the match percentage.
   - If two candidates have **similar relevance**, prioritize based on **experience and unique qualifications**.

4. **Provide a Justification for Ranking**:
   - Explain why **each candidate was ranked in their position**.
   - Highlight **strong points** and **areas where they fall short**.
   - Use a **scoring system (if applicable)** for transparency.

---

### **OUTPUT FORMAT**:
#### ✅ **Final Resume Ranking**
📌 **Rank #:** Candidate Name or Resume Summary
- **Relevance Score:** Score/100
- **Strengths:** List relevant skills, experience, certifications, projects
- **Weaknesses:** Mention missing skills, lack of experience, etc.
- **Reason for Ranking:** Explain why this candidate is the the given rank.

---

### **ADDITIONAL CONSIDERATIONS**:
- **If no resumes match the job description well**, state:
  _"None of the retrieved resumes strongly match the job description. Consider expanding the search or refining the criteria."_
- **Use professional and unbiased language**.
- **Ensure fairness** by evaluating only job-related skills and experience (avoid bias based on personal details).

---

### **EXAMPLE RESPONSE**:

📌 **Rank #1: John Doe - AI Engineer (Score: 90/100)**
- **Strengths:** 5+ years in AI development, Python, TensorFlow, relevant projects in NLP
- **Weaknesses:** No experience in cloud deployment
- **Reason for Ranking:** John has direct AI experience and matches the job description well.

📌 **Rank #2: Jane Smith - Data Scientist (Score: 85/100)**
- **Strengths:** Strong machine learning background, Python, R, SQL
- **Weaknesses:** Less hands-on AI engineering experience
- **Reason for Ranking:** Jane is a great fit but lacks AI-specific experience compared to John.

📌 **Rank #3: Mark Johnson - Software Engineer (Score: 70/100)**
- **Strengths:** Good Python skills, experience with APIs
- **Weaknesses:** No direct AI experience, lacks ML background
- **Reason for Ranking:** While Mark has strong software skills, his AI expertise is minimal.

---

### **FINAL NOTE**:
Provide a **clear and structured ranking** so recruiters can **quickly** identify the best candidates. Just give the Final ranking.
    """

    response = gemini_llm.invoke(
        [
            SystemMessage(content="You are an expert resume ranking assistant."),
            HumanMessage(content=prompt),
        ]
    )
    return response


def insert_embeddings_to_supabase(resume_texts):
    if resume_texts == "":
        resume_texts = supabase.table("resumes").select("*").execute()
        resume_texts = [resume["text"] for resume in resume_texts.data]

        # Loop through and upload
        for resume in resume_texts:
            embedding = embedding_model.embed_query(
                resume
            )  # returns List[float] of 384 dims
            supabase.table("resume_embeddings").insert(
                {"resume_text": resume, "embedding": embedding}
            ).execute()
    else:
        embedding = embedding_model.embed_query(
            resume_texts
        )  # returns List[float] of 384 dims
        supabase.table("resume_embeddings").insert(
            {"resume_text": resume_texts, "embedding": embedding}
        ).execute()


def create_embeddings(text):
    embedding = embedding_model.embed_query(text)
    return embedding
