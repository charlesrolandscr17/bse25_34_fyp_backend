from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
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
    rank_with_tfidf,
)

app = FastAPI()


class JobDescription(BaseModel):
    description: str


class Skills(BaseModel):
    user_skills: str
    job_skills: list[str]


class Ranking(BaseModel):
    job_description: str
    resumes: list[str]
    type: str


@app.post("/ranking_gemini")
async def rank_resumes(job_description: JobDescription):
    resume_texts = response = supabase.table("resumes").select("*").execute()
    resume_texts = [resume["text"] for resume in resume_texts.data]
    retriever = create_vector_store(resume_texts)
    resumes = retrieve_top_resumes(job_description.description, retriever)
    response = rank_resumes_with_gemini(job_description.description, resumes)
    return {"ranked_resumes": response}


@app.post("/ranking_tfidf")
async def rank_resumes_tfidf(ranking: Ranking):
    return rank_with_tfidf(
        ranking.job_description,
        ranking.resumes,
        ranking.type,
    )


@app.post("/recommendations")
async def get_recommendations(skills: Skills):
    print("Skills", skills)
    retriever = create_vector_store(skills.job_skills)
    recommendations = retrieve_top_resumes(skills.user_skills, retriever, top_k=3)
    return {"recommendations": recommendations}


@app.post("/create_embedding")
async def create_embeddings(text: str = ""):
    insert_embeddings_to_supabase(text)
    return {"message": "Resumes and embeddings uploaded to Supabase."}


@app.get("/get_embedding")
async def get_embeddings(query: str):
    query_embedding = create_embeddings(query)

    response = supabase.rpc(
        "match_resumes", {"query_embedding": query_embedding}
    ).execute()

    return response.data
