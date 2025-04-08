# mongodb.py
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
document_collection = db[COLLECTION_NAME]
