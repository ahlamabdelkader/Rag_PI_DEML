import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # léger et rapide

def extract_text_by_page(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({'page': i + 1, 'text': text})
    return pages

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def process_pdf_with_interest(pdf_file, interest_text):
    pages = extract_text_by_page(pdf_file)
    chunks = []
    metadata = []

    for page in pages:
        for chunk in chunk_text(page['text']):
            chunks.append(chunk)
            metadata.append({'page': page['page'], 'text': chunk})

    # Embedding des chunks
    embeddings = model.encode(chunks, convert_to_tensor=False)

    # FAISS indexation
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Embedding de la requête
    query_emb = model.encode([interest_text], convert_to_tensor=False)

    # Recherche des passages similaires
    D, I = index.search(np.array(query_emb), k=5)

    suggestions = []
    for idx in I[0]:
        suggestions.append({
            'page': metadata[idx]['page'],
            'preview': metadata[idx]['text'][:300]  # un petit aperçu
        })

    return suggestions
