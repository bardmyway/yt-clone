#!/usr/bin/env python3
"""
Build vector index from downloaded transcripts.
Uses Nomic embeddings (local) and ChromaDB.
"""

import json
from pathlib import Path
from typing import List, Dict
import tiktoken

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_nomic import NomicEmbeddings
from langchain.schema import Document

# Configuration
TRANSCRIPTS_DIR = Path("transcripts")
CHROMA_DIR = Path("chroma_db")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_transcripts() -> List[Dict]:
    """Load all transcript files with metadata."""
    transcripts = []
    
    for txt_file in TRANSCRIPTS_DIR.glob("*.txt"):
        json_file = txt_file.with_suffix(".json")
        
        if not json_file.exists():
            print(f"⚠️  Missing metadata for {txt_file.name}")
            continue
            
        # Load metadata
        with open(json_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        # Load transcript text
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        transcripts.append({
            "text": text,
            "metadata": meta
        })
    
    print(f"📚 Loaded {len(transcripts)} transcripts")
    return transcripts

def chunk_transcripts(transcripts: List[Dict]) -> List[Document]:
    """Split transcripts into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    documents = []
    
    for transcript in transcripts:
        text = transcript["text"]
        meta = transcript["metadata"]
        
        # Split into chunks
        chunks = text_splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            # Create document with metadata
            doc_meta = {
                "source": meta["video_id"],
                "title": meta["title"],
                "upload_date": meta["upload_date"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "word_count": len(chunk.split())
            }
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_meta
            ))
    
    print(f"✂️  Split into {len(documents)} chunks")
    return documents

def build_vector_store(documents: List[Document]):
    """Create ChromaDB vector store with Nomic embeddings."""
    print("🔧 Initializing Nomic embeddings...")
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    
    print("🗄️  Creating ChromaDB vector store...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="youtube_transcripts"
    )
    
    print(f"💾 Vector store saved to {CHROMA_DIR}")
    return vector_store

def main():
    print("🏗️  Building transcript index...")
    
    # Load transcripts
    transcripts = load_transcripts()
    if not transcripts:
        print("❌ No transcripts found. Run yt_transcripts.py first.")
        return
    
    # Chunk transcripts
    documents = chunk_transcripts(transcripts)
    
    # Build vector store
    vector_store = build_vector_store(documents)
    
    # Test query
    print("\n🧪 Testing with sample query...")
    results = vector_store.similarity_search("What stocks do you recommend?", k=3)
    print(f"✅ Index built successfully! Found {len(documents)} chunks.")
    print(f"📊 Sample results: {len(results)} documents retrieved for test query.")
    
    # Save index info
    index_info = {
        "total_documents": len(documents),
        "total_transcripts": len(transcripts),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": "nomic-embed-text-v1.5",
        "vector_store": "chromadb"
    }
    
    info_path = CHROMA_DIR / "index_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(index_info, f, indent=2)
    
    print(f"📋 Index info saved to {info_path}")

if __name__ == "__main__":
    main()