# -------------------------------------------------
# app.py
# -------------------------------------------------
from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings

# Use the official langchain-pinecone package (NOT the deprecated community import)
from langchain_pinecone import PineconeVectorStore

# Use Groq LLM (matches the GROQ_API_KEY in .env)
from langchain_groq import ChatGroq

# Chain imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
from src.prompt import *   # imports the `system_prompt` variable
import os

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()  # pulls PINECONE_API_KEY, GROQ_API_KEY from .env

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in your .env file")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in your .env file")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# -------------------------------------------------
# Initialise model, embeddings and vector store
# -------------------------------------------------
# 1. LLM - Groq (llama3 is fast and free on Groq)
chat_model = ChatGroq(
    model="llama-3.3-70b-versatile",   # alternatives: mixtral-8x7b-32768, llama3-70b-8192
    temperature=0.4,
    groq_api_key=GROQ_API_KEY,
)

# 2. Embeddings
embeddings = download_huggingface_embeddings()

# 3. Pinecone vector store (assumes the index already exists — run store_index.py first)
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name="medical-chatbot",   # must match the index created in store_index.py
)

# -------------------------------------------------
# Prompt template
# -------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),   # defined in src/prompt.py
        ("human", "{input}"),
    ]
)

# -------------------------------------------------
# Build the retrieval / QA chain
# -------------------------------------------------
stuff_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(
    retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
    combine_docs_chain=stuff_chain,
)
print("[INFO] RAG chain built successfully. App is ready.")

# -------------------------------------------------
# Flask web server
# -------------------------------------------------
app = Flask(__name__)

@app.route("/")
def index():
    """Render the chat UI."""
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    """
    Called by chat.html via jQuery AJAX.
    Expects form field: msg
    Returns plain-text answer.
    """
    msg = request.form.get("msg", "").strip()
    print(f"[/get] Received message: {msg}")

    if not msg:
        return "Please type a message.", 400

    try:
        response = rag_chain.invoke({"input": msg})
        print(f"[/get] RAG response keys: {response.keys()}")
        answer = response.get("answer", "").strip()
        if not answer:
            answer = "I could not find an answer. Please try rephrasing your question."
        print(f"[/get] Answer: {answer}")
        return answer
    except Exception as e:
        print(f"[/get] ERROR: {e}")
        return f"An error occurred: {str(e)}", 500


@app.route("/query", methods=["POST"])
def query():
    """
    JSON API endpoint.
    Expected JSON payload: {"question": "Your medical question?"}
    Returns: {"answer": "..."}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    response = rag_chain.invoke({"input": question})
    answer = response.get("answer", "Sorry, I could not find an answer.")
    return jsonify({"answer": answer})


if __name__ == "__main__":
    # Run on 0.0.0.0 so the container / VM can expose it
    app.run(host="0.0.0.0", port=8080, debug=True)