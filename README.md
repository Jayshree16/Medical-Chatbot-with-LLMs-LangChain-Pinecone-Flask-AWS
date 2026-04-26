# 🏥 Medical Chatbot with LLMs, LangChain, Pinecone, Flask & Render

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1.1-black?style=flat-square&logo=flask)
![LangChain](https://img.shields.io/badge/LangChain-0.3.7-green?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-AI-orange?style=flat-square)
![Pinecone](https://img.shields.io/badge/Pinecone-VectorStore-purple?style=flat-square)
![Render](https://img.shields.io/badge/Render-Deployed-4353FF?style=flat-square&logo=render)

> An intelligent medical question-answering chatbot powered by RAG (Retrieval-Augmented Generation). Ask any medical question and get accurate, context-aware answers sourced directly from medical literature.

---

## 📌 Overview

This project is a **Retrieval-Augmented Generation (RAG)** based medical chatbot that:

- 📄 Ingests medical PDF documents and stores them as vector embeddings in **Pinecone**
- 🔍 Retrieves the most relevant chunks using **semantic similarity search**
- 🤖 Generates accurate answers using **Groq's LLaMA 3.3 70B** model via **LangChain**
- 🌐 Serves a clean chat UI via **Flask**

---

## 🏗️ Architecture

```
User Question
     │
     ▼
Flask Web Server (/get)
     │
     ▼
LangChain RAG Chain
     │
     ├──► Pinecone Vector Store (semantic search → top-k chunks)
     │
     └──► Groq LLM (llama-3.3-70b-versatile) → Answer
     │
     ▼
Chat UI Response
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq — `llama-3.3-70b-versatile` |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (384-dim) |
| **Vector Store** | Pinecone (Serverless, AWS us-east-1) |
| **RAG Framework** | LangChain 0.3.7 |
| **Web Framework** | Flask 3.1.1 |
| **PDF Loader** | PyPDF |
| **Frontend** | HTML, CSS, Bootstrap 4, jQuery |
| **Deployment** | Render (Free Tier) |

---

## 📁 Project Structure

```
Medical-Chatbot/
│
├── src/
│   ├── __init__.py
│   ├── helper.py          # PDF loading, text splitting, embeddings
│   └── prompt.py          # System prompt for the LLM
│
├── templates/
│   └── chat.html          # Chat UI
│
├── static/
│   └── style.css          # Chat UI styles
│
├── data/                  # Place your medical PDF files here
│
├── research/
│   └── trials.ipynb       # Experimentation notebook
│
├── app.py                 # Flask app + RAG chain
├── store_index.py         # PDF ingestion + Pinecone indexing
├── render.yaml            # Render deployment config
├── Procfile               # Render start command
├── requirements.txt
├── setup.py
├── .env                   # API keys (never commit this!)
└── README.md
```

---

## 🚀 How to Run

### Step 1: Clone the Repository

```bash
git clone https://github.com/Jayshree16/Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS.git
cd Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS
```

### Step 2: Create & Activate Conda Environment

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

### Step 3: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY="your-pinecone-api-key"
GROQ_API_KEY="your-groq-api-key"
```

> 🔑 Get your **Pinecone API key** at [pinecone.io](https://www.pinecone.io)
> 🔑 Get your **Groq API key** at [console.groq.com](https://console.groq.com)

### Step 5: Add Medical PDFs

Place your medical PDF files inside the `data/` folder:

```bash
mkdir data
# Copy your PDFs into the data/ directory
```

### Step 6: Build the Pinecone Index

Run this **once** to ingest your PDFs and populate the vector store:

```bash
python store_index.py
```

### Step 7: Start the App

```bash
python app.py
```

Open your browser and go to: **[http://localhost:8080](http://localhost:8080)**

---

## ⚙️ Configuration

| Parameter | Value | File |
|---|---|---|
| Pinecone Index Name | `medical-chatbot` | `store_index.py` / `app.py` |
| Embedding Model | `all-MiniLM-L6-v2` | `src/helper.py` |
| Embedding Dimension | `384` | `store_index.py` |
| LLM Model | `llama-3.3-70b-versatile` | `app.py` |
| LLM Temperature | `0.4` | `app.py` |
| Top-K Retrieval | `3` | `app.py` |
| Flask Port | `8080` | `app.py` |

---

## 🔑 Key Dependencies

```
langchain==0.3.7
langchain-core==0.3.21
langchain-community==0.3.7
langchain-groq==0.2.1
langchain-pinecone==0.2.0
langchain-huggingface==0.1.2
pinecone-client==5.0.1
sentence-transformers==3.3.1
flask==3.1.1
pypdf==5.1.0
python-dotenv==1.1.0
```

---

## ☁️ Deploy on Render (Free)

This app can be deployed for free on [Render](https://render.com). Since all vector data lives in Pinecone (cloud), no persistent disk is needed.

### Step 1: Add a `render.yaml` to your project root

```yaml
services:
  - type: web
    name: medical-chatbot
    runtime: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python app.py"
    envVars:
      - key: PINECONE_API_KEY
        sync: false
      - key: GROQ_API_KEY
        sync: false
```

### Step 2: Add a `Procfile` to your project root

```
web: python app.py
```

### Step 3: Push to GitHub

```bash
git add .
git commit -m "Add Render deployment config"
git push origin main
```

### Step 4: Deploy on Render

1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect your GitHub repository
3. Set these values:
   - **Environment:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
4. Add **Environment Variables** in the Render dashboard:
   - `PINECONE_API_KEY` → your Pinecone key
   - `GROQ_API_KEY` → your Groq key
5. Click **Deploy** ✅

> ⚠️ **Free tier note:** The app sleeps after 15 minutes of inactivity. The first request after sleep may take ~30 seconds to wake up. This is normal on Render's free plan.

---

## 🙋‍♀️ Author

**Jayshree Pawar**
- GitHub: [@Jayshree16](https://github.com/Jayshree16)
- Email: jayshreepawar1612@gmail.com

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.