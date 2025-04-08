# main.py
from fastapi import FastAPI, HTTPException, status, Body
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Optional, List
import openai
import logging
import os
from dotenv import load_dotenv
from bson import ObjectId
import uvicorn
from mongodb import document_collection
from fastapi import UploadFile, File
from app_assests.embedding import get_embedding, extract_text
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG-Based AI API with MongoDB")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI Configuration
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

# Models
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, example="What is AI?")
    max_tokens: Optional[int] = Field(100, ge=50, le=500)
    document_titles: Optional[List[str]] = Field(None, example=["RESUME.pdf"])

class DocumentIn(BaseModel):
    title: str = Field(..., example="My Doc")
    content: str = Field(..., example="This is the content of the document.")

class DocumentOut(DocumentIn):
    id: str

SYSTEM_PROMPT = "You are a helpful assistant. Answer the questions based only on context. Dont answer anything else based on out of context."

@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )

# AI Endpoint
from pymongo import DESCENDING
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Helper to fetch relevant docs based on question
def find_relevant_docs(question_embedding, top_k=3):
    docs = list(document_collection.find({}, {"embedding": 1, "title": 1, "content": 1}))
    if not docs:
        return []

    doc_embeddings = np.array([doc["embedding"] for doc in docs])
    question_vec = np.array(question_embedding).reshape(1, -1)
    similarities = cosine_similarity(doc_embeddings, question_vec).flatten()

    # Rank by similarity
    ranked_docs = sorted(
        zip(similarities, docs),
        key=lambda x: x[0],
        reverse=True
    )

    top_docs = [doc for _, doc in ranked_docs[:top_k]]
    return top_docs


@app.get("/documents/titles", response_model=List[str])
async def list_document_titles():
    try:
        return [doc["title"] for doc in document_collection.find({}, {"title": 1})]
    except Exception as e:
        logger.error(f"Title fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not fetch document titles.")


@app.post("/ask")
async def ask_question(request: QuestionRequest = Body(...)):
    try:
        logger.info(f"Processing question: {request.question[:50]}...")

        context_docs = []

        if request.document_titles:
            # Fetch documents by title
            logger.info(f"Fetching documents: {request.document_titles}")
            context_docs = list(document_collection.find(
                {"title": {"$in": request.document_titles}},
                {"title": 1, "content": 1}
            ))

            if not context_docs:
                raise HTTPException(status_code=404, detail="No matching documents found for given titles.")

        else:
            # Step 1: Embed question
            question_embedding = get_embedding(request.question)

            # Step 2: Find top matching docs by similarity
            context_docs = find_relevant_docs(question_embedding)

        # Step 3: Build context from content
        context = "\n\n".join(
            f"Document Title: {doc['title']}\nContent: {doc['content'][:1000]}"
            for doc in context_docs
        ) or "No context found."

        logger.info(f"Using {len(context_docs)} documents for context")

        # Step 4: Ask OpenAI using context
        response = openai.ChatCompletion.create(
            engine=os.getenv("AZURE_OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
            ],
            temperature=0.7,
            max_tokens=request.max_tokens
        )

        answer = response.choices[0].message['content'].strip()
        logger.info(f"Generated answer with {len(answer.split())} words")

        return {
            "answer": answer,
            "sources": [doc["title"] for doc in context_docs]
        }

    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service unavailable"
        )

# Document Upload
@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    supported_exts = (".pdf", ".docx", ".xlsx", ".xls")
    if not file.filename.lower().endswith(supported_exts):
        raise HTTPException(status_code=400, detail="Supported formats: PDF, DOCX, XLSX")

    try:
        # Save file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Extract text & embed
        text = extract_text(temp_path)
        embedding = get_embedding(text)

        doc = {
            "title": file.filename,
            "content": text,
            "embedding": embedding
        }

        result = document_collection.insert_one(doc)
        return {"id": str(result.inserted_id), "title": file.filename}

    except Exception as e:
        logger.error(f"Upload Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.get("/documents/", response_model=List[DocumentOut])
async def get_documents():
    try:
        docs = []
        for doc in document_collection.find():
            docs.append(DocumentOut(
                id=str(doc.get("_id")),
                title=doc.get("title", "No Title"),
                content=doc.get("content", "No Content")
            ))
        return docs
    except Exception as e:
        logger.error(f"Read failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")

# Health Check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.1"}

