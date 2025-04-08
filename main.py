from fastapi import FastAPI, HTTPException, status, Body, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Optional, List
from openai import AzureOpenAI
import logging
import os
from dotenv import load_dotenv
from bson import ObjectId
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from jose import JWTError, jwt
import bcrypt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from mongodb import document_collection, user_collection
from app_assests.embedding import get_embedding, extract_text

# Load env vars
load_dotenv()

# Logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(title="RAG-Based AI API with MongoDB + Auth")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pdf-reader-ai-4d5a2.web.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI Client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# JWT Config
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Models
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3)
    max_tokens: Optional[int] = Field(100, ge=50, le=500)
    document_titles: Optional[List[str]] = None

class DocumentIn(BaseModel):
    title: str
    content: str

class DocumentOut(DocumentIn):
    id: str

class UserCreate(BaseModel):
    username: str = Field(..., example="john_doe")
    password: str = Field(..., min_length=6)

class Token(BaseModel):
    access_token: str
    token_type: str

SYSTEM_PROMPT = "You are a helpful assistant. Answer the questions based only on context. Don't answer anything outside the given context."

# Utility functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=401, detail="Invalid token")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        user = user_collection.find_one({"username": username})
        if not user:
            raise credentials_exception
        return user
    except JWTError:
        raise credentials_exception

@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# Register
@app.post("/register", status_code=201)
async def register(user: UserCreate):
    if user_collection.find_one({"username": user.username}):
        raise HTTPException(400, detail="Username already exists")
    user_collection.insert_one({"username": user.username, "password": hash_password(user.password)})
    return {"message": "User registered"}

# Login
@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = user_collection.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(400, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

# Helper for doc retrieval
def find_relevant_docs(question_embedding, top_k=3):
    docs = list(document_collection.find({}, {"embedding": 1, "title": 1, "content": 1}))
    if not docs:
        return []
    doc_embeddings = np.array([doc["embedding"] for doc in docs])
    question_vec = np.array(question_embedding).reshape(1, -1)
    similarities = cosine_similarity(doc_embeddings, question_vec).flatten()
    ranked_docs = sorted(zip(similarities, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked_docs[:top_k]]

# Ask AI
@app.post("/ask")
async def ask_question(request: QuestionRequest = Body(...), current_user: dict = Depends(get_current_user)):
    try:
        logger.info(f"Processing question: {request.question}")
        if request.document_titles:
            context_docs = list(document_collection.find(
                {"title": {"$in": request.document_titles}}, {"title": 1, "content": 1}
            ))
            if not context_docs:
                raise HTTPException(404, detail="No matching documents found.")
        else:
            question_embedding = get_embedding(request.question)
            context_docs = find_relevant_docs(question_embedding)

        context = "\n\n".join(
            f"Document Title: {doc['title']}\nContent: {doc['content'][:1000]}"
            for doc in context_docs
        ) or "No context found."

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
            ],
            temperature=0.7,
            max_tokens=request.max_tokens
        )

        answer = response.choices[0].message.content.strip()
        return {"answer": answer, "sources": [doc["title"] for doc in context_docs]}
    except AzureOpenAI.error.OpenAIError as e:
        logger.error(f"OpenAI Error: {str(e)}")
        raise HTTPException(503, detail="AI service unavailable")

# Upload Document
@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    supported_exts = (".pdf", ".docx", ".xlsx", ".xls")
    if not file.filename.lower().endswith(supported_exts):
        raise HTTPException(400, detail="Supported formats: PDF, DOCX, XLSX")
    try:
        content = await file.read()
        text = extract_text(content, file.filename)
        embedding = get_embedding(text)
        doc = {
            "title": file.filename,
            "content": text,
            "embedding": embedding
        }
        result = document_collection.insert_one(doc)
        return {"id": str(result.inserted_id), "title": file.filename}
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(500, detail=f"Upload failed: {str(e)}")

@app.get("/documents/titles", response_model=List[str])
async def list_document_titles():
    try:
        return [doc["title"] for doc in document_collection.find({}, {"title": 1})]
    except Exception as e:
        logger.error(f"Title fetch failed: {str(e)}")
        raise HTTPException(500, detail="Failed to fetch titles")

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
        raise HTTPException(500, detail="Failed to fetch documents")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.1"}
