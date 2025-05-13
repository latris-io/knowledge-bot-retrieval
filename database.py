# database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Load DB credentials from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# Set up the SQLAlchemy engine with SSL (e.g., for Render Postgres)
engine = create_engine(DATABASE_URL, connect_args={"sslmode": "require"})

# Create a session factory
SessionLocal = sessionmaker(bind=engine)
