#!/usr/bin/env python3
"""
Embedding and Indexing Module for Amharic Labour Proclamation No. 1156/2019.
Embeds chunks with intfloat/multilingual-e5-base for hybrid vector search / FAISS.

Usage:
  python -m src.embed --build
"""
import os
import sys
import json
import time
import pickle
import hashlib
import argparse
from pathlib import Path
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.fidel_fold import fold

MODEL_NAME = "intfloat/multilingual-e5-base"

def mean_pooling(model_output, attention_mask):
    """Perform mean pooling taking attention mask into account."""
    import torch
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_texts(texts, model_name=MODEL_NAME, batch_size=32, device=None):
    """
    Generate normalized embeddings for list of texts using SentenceTransformers / PyTorch.
    """
    import torch
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"[embed_texts] Loading {model_name} on {device}...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype("float32")
    except Exception:
        from transformers import AutoTokenizer, AutoModel
        import torch.nn.functional as F
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)
        model.eval()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = model(**encoded_input)
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                all_embeddings.append(sentence_embeddings.cpu().numpy())
        return np.vstack(all_embeddings).astype("float32")

def build_index(
    chunks_path: Path = None,
    index_dir: Path = None,
    model_name: str = MODEL_NAME
):
    """
    Builds full retrieval index (embeddings.npy, chunks.faiss, chunks.json, tfidf.pkl, build_info.json)
    using intfloat/multilingual-e5-base (768 dim).
    """
    data_dir = BASE_DIR / 'data'
    if chunks_path is None:
        chunks_path = data_dir / 'chunks.json'
    if index_dir is None:
        index_dir = BASE_DIR / 'index'
    index_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    # Format passages for E5 model: "passage: <text_folded>"
    formatted_passages = ["passage: " + c.get('text_folded', fold(c.get('text_spoken', ''))) for c in chunks]
    print(f"Generating E5-base embeddings for {len(formatted_passages)} chunks...")
    embeddings = embed_texts(formatted_passages, model_name=model_name)
    dim = embeddings.shape[1]
    print(f"Generated embeddings matrix shape: {embeddings.shape} (Dimension: {dim})")
    
    # 1. Save embeddings.npy
    emb_file = index_dir / 'embeddings.npy'
    np.save(emb_file, embeddings)
    print(f"Saved {emb_file}")
    
    # 2. Save chunks.json
    chunks_file = index_dir / 'chunks.json'
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {chunks_file}")
    
    # 3. Build & save FAISS Index
    try:
        import faiss
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss_file = index_dir / 'chunks.faiss'
        faiss.write_index(index, str(faiss_file))
        print(f"Saved FAISS index to {faiss_file} (Total: {index.ntotal})")
    except ImportError:
        print("FAISS not installed locally; FAISS index will be built on runtime/Colab.")
        
    # 4. Build & save TF-IDF Vectorizer
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=1)
        doc_texts = [c.get("text_folded", fold(c.get("text_spoken", ""))) for c in chunks]
        tfidf.fit(doc_texts)
        tfidf_file = index_dir / 'tfidf.pkl'
        with open(tfidf_file, 'wb') as f:
            pickle.dump(tfidf, f)
        print(f"Saved TF-IDF model to {tfidf_file}")
    except Exception as e:
        print(f"Warning: Could not build TF-IDF: {e}")
        
    # 5. Save build_info.json
    with open(chunks_file, 'rb') as f:
        chunks_hash = hashlib.sha256(f.read()).hexdigest()
        
    build_info = {
        "model": model_name,
        "num_chunks": len(chunks),
        "dim": dim,
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "normalized": True,
        "index_type": "IndexFlatIP",
        "chunks_sha256": chunks_hash,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    info_file = index_dir / 'build_info.json'
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(build_info, f, indent=2)
    print(f"Saved {info_file}")
    print("\n✅ Build complete! Index is ready with intfloat/multilingual-e5-base (dim=768).")

def main():
    parser = argparse.ArgumentParser(description="Embed chunks with intfloat/multilingual-e5-base")
    parser.add_argument('--build', action='store_true', help="Build full index in index/ directory")
    parser.add_argument('--chunks', default=None, help="Path to data/chunks.json")
    parser.add_argument('--model', default=MODEL_NAME, help="E5 Model ID")
    args = parser.parse_args()
    
    if args.build or args.chunks:
        chunks_path = Path(args.chunks) if args.chunks else None
        build_index(chunks_path=chunks_path, model_name=args.model)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
