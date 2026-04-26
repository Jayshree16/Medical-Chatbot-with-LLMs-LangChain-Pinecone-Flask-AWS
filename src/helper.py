# src/helper.py
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def load_pdf_files(data_dir: str):
    """
    Recursively load all PDF files from `data_dir`.
    Returns a list of LangChain Document objects.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Make sure you have a 'data/' folder with your PDF files."
        )

    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    print(f"[helper] Loaded {len(documents)} pages from PDFs in '{data_dir}'")
    return documents


def text_split(extracted_data, chunk_size: int = 500, chunk_overlap: int = 20):
    """
    Split documents into smaller chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(extracted_data)
    print(f"[helper] Split into {len(chunks)} chunks "
          f"(chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def download_huggingface_embeddings():
    """
    Load the all-MiniLM-L6-v2 sentence-transformer model (384-dim).
    This must match the `dimension` set when creating the Pinecone index.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("[helper] HuggingFace embeddings loaded (all-MiniLM-L6-v2, 384-dim)")
    return embeddings