from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langdetect import detect
import requests
import os

# Initialisation FastAPI
app = FastAPI()

# CORS pour le frontend (accès depuis n’importe quel domaine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle d’embedding léger
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Appel API HuggingFace (token à configurer dans Render > Environment)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

def call_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 300}
    }
    response = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        headers=headers,
        json=payload
    )
    if response.status_code == 200:
        result = response.json()
        return result[0]["generated_text"].split("Réponse:")[-1].strip()
    else:
        return "Erreur LLM : " + response.text

def clean_text(text):
    return ' '.join([line.strip() for line in text.split('\n') if len(line.strip()) > 5])

def segment_text(text, max_chars=500):
    sentences = text.split('.')
    chunks, chunk = [], ""
    for s in sentences:
        if len(chunk) + len(s) < max_chars:
            chunk += " " + s
        else:
            chunks.append(chunk.strip())
            chunk = s
    if chunk:
        chunks.append(chunk.strip())
    return chunks

@app.post("/query/")
async def query(pdf: UploadFile = File(...), question: str = Form(...)):
    doc = fitz.open(stream=await pdf.read(), filetype="pdf")
    full_text = "\n".join([page.get_text() for page in doc])
    lang = detect(full_text)

    chunks = []
    for page_num, page in enumerate(doc, 1):
        page_text = page.get_text()
        raw_chunks = segment_text(page_text)
        for ch in raw_chunks:
            cleaned = clean_text(ch)
            if len(cleaned.split()) >= 5:
                chunks.append({"chunk": cleaned, "page": page_num})

    texts = [c["chunk"] for c in chunks]
    emb = embedding_model.encode(texts)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    q_vec = embedding_model.encode([question])
    q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)
    _, I = index.search(q_vec, 3)
    context_chunks = [chunks[i] for i in I[0]]

    context_text = "\n\n".join([c["chunk"] for c in context_chunks])
    prompt = f"Contexte:\n{context_text}\n\nQuestion: {question}\nRéponse:"
    answer = call_llm(prompt)

    return {
        "answer": answer,
        "context": context_chunks
    }
