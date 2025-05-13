# knowledge-bot-ingestion-service/bot_config.py

import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("config_debug.log")
    ]
)
logger = logging.getLogger(__name__)

# Load .env file only in local/dev environments
load_dotenv()

def get_openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        logger.error("[CONFIG] OPENAI_API_KEY not found in environment")
        raise ValueError("OPENAI_API_KEY not found in environment")
    logger.info("[CONFIG] Successfully loaded OPENAI_API_KEY")
    return key

def get_db_url():
    url = os.getenv("DATABASE_URL")
    if not url:
        logger.error("[CONFIG] DATABASE_URL not found in environment")
        raise ValueError("DATABASE_URL not found in environment")
    logger.info("[CONFIG] Successfully loaded DATABASE_URL")
    return url

def get_bucket_name():
    bucket = os.getenv("AWS_BUCKET_NAME")
    if not bucket:
        logger.error("[CONFIG] AWS_BUCKET_NAME not found in environment")
        raise ValueError("AWS_BUCKET_NAME not found in environment")
    logger.info("[CONFIG] Successfully loaded AWS_BUCKET_NAME")
    return bucket

def get_s3_client():
    import boto3
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_access_secret = os.getenv("AWS_ACCESS_SECRET")
    if not all([aws_access_key_id, aws_access_secret]):
        logger.error("[CONFIG] AWS_ACCESS_KEY_ID or AWS_ACCESS_SECRET not found in environment")
        raise ValueError("AWS_ACCESS_KEY_ID or AWS_ACCESS_SECRET not found in environment")
    client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_access_secret,
    )
    logger.info("[CONFIG] Successfully initialized S3 client")
    return client

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
logger.info(f"[CONFIG] CHROMA_DIR set to {CHROMA_DIR}")