#!/usr/bin/env python3
"""
Chat with a YouTube creator clone.
Uses vector search to find relevant content, then prompts an LLM.
"""

import json
from pathlib import Path
import sys

from langchain_chroma import Chroma
from langchain_nomic import NomicEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

# Configuration
CHROMA_DIR = Path("chroma_db")
TRANSCRIPTS_DIR = Path("transcripts")
MODEL = "llama3.2"  # Ollama model
# Or use OpenAI: model = ChatOpenAI(model="gpt-4-turbo")

def load_creator_info() -> str:
    """Extract creator name from transcripts."""
    summary_path = TRANSCRIPTS_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        channel_url = summary.get("channel_url", "Unknown Creator")
        # Extract channel name from URL
        if "@" in channel_url:
            channel_name = channel_url.split("@")[1].split("/")[0]
            return channel_name.replace("-", " ").title()
    return "The Creator"

def initialize_chain():
    """Initialize the RAG chain."""
    print("🔧 Loading vector store...")
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="youtube_transcripts"
    )
    
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )
    
    print(f"🤖 Initializing LLM ({MODEL})...")
    llm = Ollama(model=MODEL)
    # For OpenAI: llm = ChatOpenAI(model="gpt-4-turbo")
    
    creator_name = load_creator_info()
    
    # System prompt
    template = """You are {creator_name}, a YouTube creator. You only answer questions using knowledge from your actual videos.

Relevant excerpts from your videos:
{context}

Question: {question}

Answer in {creator_name}'s style, using only the information above. If you don't know, say "I haven't discussed that in my videos."
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join([f"• {doc.page_content}" for doc in docs])
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough(), "creator_name": lambda x: creator_name}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, creator_name

def main():
    print("🤖 YouTube Creator Chat")
    print("=" * 40)
    
    # Check if index exists
    if not CHROMA_DIR.exists():
        print("❌ Vector index not found. Run build_index.py first.")
        return
    
    chain, creator_name = initialize_chain()
    
    print(f"\n💬 Chatting with {creator_name}")
    print("Type 'quit' to exit, 'clear' to start new conversation")
    print("-" * 40)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                conversation_history = []
                print("🧹 Conversation cleared")
                continue
            elif not user_input:
                continue
            
            print(f"\n{creator_name}: ", end="", flush=True)
            
            # Invoke chain
            response = chain.invoke(user_input)
            print(response)
            
            # Store in history
            conversation_history.append({
                "user": user_input,
                "assistant": response
            })
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()