# app/services/logger.py
import pymongo
from app.config import MONGO_URI

client = pymongo.MongoClient(MONGO_URI)
db = client["resume_analyzer"]
collection = db["logs"]


def save_log(log_data: dict):
    collection.insert_one(log_data)
