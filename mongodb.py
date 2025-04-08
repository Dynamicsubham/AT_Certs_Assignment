# mongodb.py
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()

# Environment config
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME")
DOCUMENT_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_Document")
USER_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_Users")

# Database connection
client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

# Collections
document_collection = db[DOCUMENT_COLLECTION_NAME]
user_collection = db[USER_COLLECTION_NAME]
