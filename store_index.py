from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec  # pinecone-client v5+
from langchain_pinecone import PineconeVectorStore
from src.helper import (
    load_pdf_files,
    text_split,
    download_huggingface_embeddings,
)

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in your .env file")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# -------------------------------------------------
# Load and split PDF data
# -------------------------------------------------
# Use a relative path so the project works on any machine
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

print(f"[INFO] Loading PDFs from: {DATA_DIR}")
extracted_data = load_pdf_files(DATA_DIR)

print(f"[INFO] Splitting documents into chunks...")
text_chunks = text_split(extracted_data)
print(f"[INFO] Total chunks created: {len(text_chunks)}")

# -------------------------------------------------
# Embeddings
# -------------------------------------------------
print("[INFO] Loading HuggingFace embeddings...")
embeddings = download_huggingface_embeddings()

# -------------------------------------------------
# Pinecone setup
# -------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "medical-chatbot"
DIMENSION = 384       # must match all-MiniLM-L6-v2 output
METRIC = "cosine"

# Create index only if it doesn't already exist
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"[INFO] Creating Pinecone index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"[INFO] Index '{INDEX_NAME}' created successfully.")
else:
    print(f"[INFO] Index '{INDEX_NAME}' already exists. Skipping creation.")

# -------------------------------------------------
# Upsert documents into Pinecone
# -------------------------------------------------
print("[INFO] Upserting document chunks into Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=INDEX_NAME,
)
print("[INFO] All chunks upserted successfully. Index is ready.")