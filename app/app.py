from typing import Union
from fastapi import FastAPI
from supabase import create_client, Client
from dotenv import dotenv_values

config = dotenv_values(".env")

SUPABASE_URL = config["SUPABASE_URL"]
SUPABASE_KEY = config["SUPABASE_ANON_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

from resume_ranker.resume_ranker import (
    create_embeddings,
    insert_embeddings_to_supabase,
    retrieve_top_resumes,
    rank_resumes_with_gemini,
    create_vector_store,
)

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "word"}


@app.post("/ranking")
async def rank_resumes(job_description: str):
    resume_texts = response = supabase.table("resumes").select("*").execute()
    resume_texts = [resume["text"] for resume in resume_texts.data]
    retriever = create_vector_store(resume_texts)
    resumes = retrieve_top_resumes(job_description, retriever)
    response = rank_resumes_with_gemini(job_description, resumes)
    return {"ranked_resumes": response}


@app.post("/recommendations")
async def get_recommendations(user_skills: str):
    job_skills = [
        "SQL, CSS, AI, JavaScript, Data Science",
        "AI, Data Science, SQL, Python, CSS",
        "SQL, AI, Python",
        "Java, AI, Python, Data Science, Machine Learning",
        "Machine Learning, C++",
    ]

    retriever = create_vector_store(job_skills)
    recommendations = retrieve_top_resumes(user_skills, retriever, top_k=3)
    return {"recommendations": recommendations}


@app.post("/create_embedding")
async def test_api(text: str = ""):
    insert_embeddings_to_supabase(text)
    return {"message": "Resumes and embeddings uploaded to Supabase."}


@app.get("/get_embedding")
async def get_embeddings(query: str):
    query_embedding = create_embeddings(query)

    response = supabase.rpc(
        "match_resumes", {"query_embedding": query_embedding}
    ).execute()

    return response.data
