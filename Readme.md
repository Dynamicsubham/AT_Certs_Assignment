# RAG-Based AI FAQ System with FastAPI, MongoDB, Azure OpenAI & JWT Auth

## 🧠 Overview
This project is a **Retrieval-Augmented Generation (RAG)**-powered AI system that allows users to upload documents, extract embeddings using Azure OpenAI, and ask questions based on their contents. It features **user authentication**, document management, and semantic search for context-aware Q&A.

## 🔧 Tech Stack

| Component            | Tech Used                                     |
|---------------------|-----------------------------------------------|
| Backend Framework   | FastAPI                                       |
| Authentication      | JWT (OAuth2PasswordBearer)                    |
| Vector Embedding    | Azure OpenAI (`text-embedding-3-large`)       |
| LLM Completion      | Azure OpenAI Chat Completion API              |
| Document Storage    | MongoDB (documents + vector embeddings)       |
| Frontend Hosting    | Firebase Web Hosting                          |
| File Parsing        | PyMuPDF / python-docx / pandas (via `extract_text`) |
| Deployment          | Docker, Render                                |

---

## 📁 Features

- 🔐 User registration & login using JWT tokens
- 📄 Upload and parse `.pdf`, `.docx`, and `.xlsx` files
- 📚 Generate vector embeddings for documents and store in MongoDB
- ❓ Ask questions — relevant document contexts are retrieved using cosine similarity
- 💬 Get AI-generated answers via Azure OpenAI Chat API
- 🧠 Smart retrieval via embedding similarity search

---

## 🏗️ Software Architecture

![Software Architecture](https://github.com/Dynamicsubham/AT_Certs_Assignment/blob/master/Fast%20API%20Backend.png)
---

## 🚀 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/rag-fastapi-mongo-auth.git
cd rag-fastapi-mongo-auth
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file:

```env
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_MODEL=text-embedding-3-large
SECRET_KEY=your_jwt_secret_key
MONGO_URI=mongodb+srv://...your-uri...
```

### 4. Run the App
```bash
uvicorn main:app --reload
```

---

## 🔐 Auth Flow
- `POST /register` — register a new user
- `POST /token` — obtain JWT token via username + password
- Authenticated endpoints:
  - `POST /documents/upload`
  - `POST /ask`

Use the JWT token as `Authorization: Bearer <token>` in requests.

---

## 📘 API Endpoints

### Authentication
| Method | Endpoint     | Description              |
|--------|--------------|--------------------------|
| POST   | `/register`  | Register new user        |
| POST   | `/token`     | Login & get access token |

### Document Management
| Method | Endpoint               | Description                  |
|--------|------------------------|------------------------------|
| POST   | `/documents/upload`    | Upload & embed new document |
| GET    | `/documents/`          | Get all stored documents     |
| GET    | `/documents/titles`    | Get document titles only     |

### Question Answering
| Method | Endpoint  | Description                       |
|--------|-----------|-----------------------------------|
| POST   | `/ask`    | Ask a question (JWT protected)    |

---

## 🧪 Testing the API
Use [Postman](https://www.postman.com/) or [Hoppscotch](https://hoppscotch.io/) with these steps:

1. Register a user: `POST /register`
2. Login: `POST /token` to get JWT
3. Upload PDF: `POST /documents/upload` with file (authenticated)
4. Ask: `POST /ask` with question and optional document titles

---

## 📦 Deployment
This backend can be deployed on Render or any Docker-compatible cloud:

```bash
docker build -t rag-fastapi .
docker run -p 8000:8000 rag-fastapi
```

---

## 🔍 Project Structure

```
├── main.py                # FastAPI app with routes and logic
├── mongodb.py             # MongoDB client and collections
├── app_assests/
│   └── embedding.py       # extract_text & embedding logic
├── requirements.txt       # Dependencies
├── .env                   # Environment variables
└── README.md
```

---

## 📄 License
MIT License. Use freely and contribute if you'd like 💙

---

## 🙌 Acknowledgments
- OpenAI for embeddings & completions
- FastAPI for rapid backend development
- MongoDB for flexible document storage
- Firebase for seamless frontend hosting

---

## 🔗 Live Frontend
🌐 [pdf-reader-ai-4d5a2.web.app](https://pdf-reader-ai-4d5a2.web.app)

## 📡 Backend Endpoint
🧠 [`https://at-certs-assignment.onrender.com`](https://at-certs-assignment.onrender.com)

