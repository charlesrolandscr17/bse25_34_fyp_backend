# from typing import Union
# from fastapi import FastAPI
# from pydantic import BaseModel
# from supabase import create_client, Client
# from dotenv import dotenv_values
# from fastapi import FastAPI
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer, util


# # Create FastAPI app
# app = FastAPI(title="Resume Matcher API", version="1.0")

# ranking_model = SentenceTransformer("scr17/fyp")

# # config = dotenv_values("../.env")

# # SUPABASE_URL = config["SUPABASE_URL"]
# # SUPABASE_KEY = config["SUPABASE_ANON_KEY"]

# # supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# # from resume_ranker.resume_ranker import (
# #     create_embeddings,
# #     insert_embeddings_to_supabase,
# #     retrieve_top_resumes,
# #     rank_resumes_with_gemini,
# #     create_vector_store,
# #     rank_with_tfidf,
# # )

# # app = FastAPI()


# # class JobDescription(BaseModel):
# #     description: str


# # class Skills(BaseModel):
# #     user_skills: str
# #     job_skills: list[str]


# # class Ranking(BaseModel):
# #     job_description: str
# #     resumes: list[str]
# #     type: str


# # @app.post("/ranking_gemini")
# # async def rank_resumes(job_description: JobDescription):
# #     resume_texts = response = supabase.table("resumes").select("*").execute()
# #     resume_texts = [resume["text"] for resume in resume_texts.data]
# #     retriever = create_vector_store(resume_texts)
# #     resumes = retrieve_top_resumes(job_description.description, retriever)
# #     response = rank_resumes_with_gemini(job_description.description, resumes)
# #     return {"ranked_resumes": response}


# # @app.post("/ranking_tfidf")
# # async def rank_resumes_tfidf(ranking: Ranking):
# #     return rank_with_tfidf(
# #         ranking.job_description,
# #         ranking.resumes,
# #         ranking.type,
# #     )


# # @app.post("/recommendations")
# # async def get_recommendations(skills: Skills):
# #     print("Skills", skills)
# #     retriever = create_vector_store(skills.job_skills)
# #     recommendations = retrieve_top_resumes(skills.user_skills, retriever, top_k=3)
# #     return {"recommendations": recommendations}


# # @app.post("/create_embedding")
# # async def create_embeddings(text: str = ""):
# #     insert_embeddings_to_supabase(text)
# #     return {"message": "Resumes and embeddings uploaded to Supabase."}


# # @app.get("/get_embedding")
# # async def get_embeddings(query: str):
# #     query_embedding = create_embeddings(query)

# #     response = supabase.rpc(
# #         "match_resumes", {"query_embedding": query_embedding}
# #     ).execute()

# #     return response.data


# # @app.get("/")
# # async def test():
# #     return {"test": "Hello world"}


# # Define request schema
# class MatchRequest(BaseModel):
#     job_description: str
#     resume: str
#     threshold: float = 0.75  # Optional threshold (default is 0.75)


# # Define response schema
# class MatchResponse(BaseModel):
#     similarity_score: float
#     decision: str

# @app.post("/match", response_model=MatchResponse)
# def match_resumes(req: MatchRequest):
#     # Encode inputs
#     jd_emb = ranking_model.encode(req.job_description, convert_to_tensor=True)
#     cv_emb = ranking_model.encode(req.resume, convert_to_tensor=True)

#     # Cosine similarity
#     score = util.cos_sim(jd_emb, cv_emb).item()
#     decision = "select" if score >= req.threshold else "reject"

#     return MatchResponse(similarity_score=round(score, 4), decision=decision)

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from parser.extract_text import get_text

# Initialize FastAPI
app = FastAPI(
    title="Resume Matcher API", version="3.0 (with Stopwords + Punctuation Removal)"
)

# Load your fine-tuned SBERT model
model = SentenceTransformer("scr17/fyp")

# Download NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Load stopwords
stop_words = set(stopwords.words("english"))


# Cleaning function
def clean_text(text: str) -> str:
    # Tokenize
    tokens = word_tokenize(text)

    # Remove punctuation and stopwords
    cleaned_tokens = [
        word
        for word in tokens
        if word.lower() not in stop_words and word not in string.punctuation
    ]

    # Join back into string
    return " ".join(cleaned_tokens)


# Pydantic request and response schemas
class MatchRequest(BaseModel):
    job_description: str
    resume: str
    threshold: float = 0.75


class WordMatch(BaseModel):
    resume_word: str
    best_job_word: str
    score: float


class MatchResponse(BaseModel):
    similarity_score: float
    decision: str
    word_matches: list[WordMatch]


# Main matching endpoint
@app.post("/match", response_model=MatchResponse)
def match_resumes(req: MatchRequest):
    # Preprocess inputs
    cleaned_job_desc = clean_text(req.job_description)
    cleaned_resume = clean_text(req.resume)

    # Encode full inputs
    jd_embedding = model.encode(cleaned_job_desc, convert_to_tensor=True)
    cv_embedding = model.encode(cleaned_resume, convert_to_tensor=True)

    # Compute overall similarity
    score = util.cos_sim(jd_embedding, cv_embedding).item()
    decision = "select" if score >= req.threshold else "reject"

    # Tokenize cleaned inputs
    job_tokens = cleaned_job_desc.split()
    resume_tokens = cleaned_resume.split()

    # Encode tokens individually
    job_embeddings = model.encode(job_tokens, convert_to_tensor=True)
    resume_embeddings = model.encode(resume_tokens, convert_to_tensor=True)

    # Compute pairwise similarity matrix
    similarity_matrix = util.cos_sim(job_embeddings, resume_embeddings)

    # For each resume word, find best matching job word
    word_matches = []
    for i, resume_token in enumerate(resume_tokens):
        best_match_idx = similarity_matrix[:, i].argmax()
        best_match_score = similarity_matrix[best_match_idx, i].item()
        match = WordMatch(
            resume_word=resume_token,
            best_job_word=job_tokens[best_match_idx],
            score=round(best_match_score, 4),
        )
        word_matches.append(match)

    return MatchResponse(
        similarity_score=round(score, 4), decision=decision, word_matches=word_matches
    )


@app.post("/match_files", response_model=MatchResponse)
def match_resumes(req: MatchRequest):
    # Preprocess inputs
    cleaned_job_desc = clean_text(get_text(req.job_description))
    cleaned_resume = clean_text(get_text(req.resume))

    # Encode full inputs
    jd_embedding = model.encode(cleaned_job_desc, convert_to_tensor=True)
    cv_embedding = model.encode(cleaned_resume, convert_to_tensor=True)

    # Compute overall similarity
    score = util.cos_sim(jd_embedding, cv_embedding).item()
    decision = "select" if score >= req.threshold else "reject"

    # Tokenize cleaned inputs
    job_tokens = cleaned_job_desc.split()
    resume_tokens = cleaned_resume.split()

    # Encode tokens individually
    job_embeddings = model.encode(job_tokens, convert_to_tensor=True)
    resume_embeddings = model.encode(resume_tokens, convert_to_tensor=True)

    # Compute pairwise similarity matrix
    similarity_matrix = util.cos_sim(job_embeddings, resume_embeddings)

    # For each resume word, find best matching job word
    word_matches = []
    for i, resume_token in enumerate(resume_tokens):
        best_match_idx = similarity_matrix[:, i].argmax()
        best_match_score = similarity_matrix[best_match_idx, i].item()
        match = WordMatch(
            resume_word=resume_token,
            best_job_word=job_tokens[best_match_idx],
            score=round(best_match_score, 4),
        )
        word_matches.append(match)

    return MatchResponse(
        similarity_score=round(score, 4), decision=decision, word_matches=word_matches
    )


@app.post("/match_resume_file", response_model=MatchResponse)
def match_resumes(req: MatchRequest):
    # Preprocess inputs
    cleaned_job_desc = clean_text(get_text(req.job_description))
    cleaned_resume = clean_text(get_text(req.resume))

    # Encode full inputs
    jd_embedding = model.encode(cleaned_job_desc, convert_to_tensor=True)
    cv_embedding = model.encode(cleaned_resume, convert_to_tensor=True)

    # Compute overall similarity
    score = util.cos_sim(jd_embedding, cv_embedding).item()
    decision = "select" if score >= req.threshold else "reject"

    # Tokenize cleaned inputs
    job_tokens = cleaned_job_desc.split()
    resume_tokens = cleaned_resume.split()

    # Encode tokens individually
    job_embeddings = model.encode(job_tokens, convert_to_tensor=True)
    resume_embeddings = model.encode(resume_tokens, convert_to_tensor=True)

    # Compute pairwise similarity matrix
    similarity_matrix = util.cos_sim(job_embeddings, resume_embeddings)

    # For each resume word, find best matching job word
    word_matches = []
    for i, resume_token in enumerate(resume_tokens):
        best_match_idx = similarity_matrix[:, i].argmax()
        best_match_score = similarity_matrix[best_match_idx, i].item()
        match = WordMatch(
            resume_word=resume_token,
            best_job_word=job_tokens[best_match_idx],
            score=round(best_match_score, 4),
        )
        word_matches.append(match)

    return MatchResponse(
        similarity_score=round(score, 4), decision=decision, word_matches=word_matches
    )
