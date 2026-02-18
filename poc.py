#!/usr/bin/env python3
"""
YT Clone - POC with optional Vector DB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Minimum:  pip install yt-dlp
+ Vector: pip install chromadb  (auto-detected, falls back to keyword search)
+ AI:     pip install ollama  (then: ollama pull llama3.2)

Usage:
  python3 poc.py "https://www.youtube.com/@NolanGouveia/videos" 30
  python3 poc.py "https://youtu.be/VIDEO_ID" 14

No OpenAI. No API keys. Works 100% offline.
"""

import subprocess, sys, os, re
from datetime import datetime, timedelta


# ─── Transcript Download ─────────────────────────────────────────────────────

def download_transcripts(channel_url, days=30):
    after = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    os.makedirs("transcripts", exist_ok=True)

    print(f"📥 Fetching last {days} days of transcripts...")
    subprocess.run([
        "yt-dlp",
        "--write-auto-subs", "--skip-download",
        "--sub-langs", "en",
        "--sub-format", "vtt",
        "--dateafter", after,
        "--playlist-end", "30",
        "--quiet", "--no-warnings",
        "-o", "transcripts/%(title)s.%(ext)s",
        channel_url
    ])

    docs = []
    for f in sorted(os.listdir("transcripts")):
        if f.endswith(".en.vtt"):
            text = vtt_to_text(f"transcripts/{f}")
            title = f.replace(".en.vtt", "")
            if len(text) > 100:
                docs.append((title, text))
                print(f"  ✓ {title[:70]}")

    return docs


def vtt_to_text(path):
    """Strip ALL VTT markup — clean readable text only"""
    with open(path, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Strip inline timestamp tags: <00:00:31.119> <c> </c> and any HTML tags
    content = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d+>', '', content)
    content = re.sub(r'</?c>', '', content)
    content = re.sub(r'<[^>]+>', '', content)

    lines, seen = [], set()
    for line in content.splitlines():
        line = line.strip()
        if not line: continue
        if re.match(r"^\d{2}:\d{2}.*-->", line): continue  # timestamp line
        if line in ("WEBVTT", "Kind: captions", "Kind: subtitles"): continue
        if line.startswith(("Language:", "NOTE", "align:", "position:")): continue
        if re.match(r"^\d+$", line): continue               # cue number
        if line not in seen:
            seen.add(line)
            lines.append(line)

    return " ".join(lines)


# ─── Vector DB (optional ChromaDB) ───────────────────────────────────────────

def build_vector_index(docs):
    """Build ChromaDB index. Returns collection or None if unavailable."""
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        return None

    print("\n🗄️  Building vector index (ChromaDB)...")
    client = chromadb.Client()
    ef = embedding_functions.DefaultEmbeddingFunction()
    col = client.get_or_create_collection("yt_clone", embedding_function=ef)

    for i, (title, text) in enumerate(docs):
        # Split long transcripts into chunks
        chunks = [text[j:j+800] for j in range(0, len(text), 600)]
        col.add(
            documents=chunks,
            ids=[f"doc_{i}_chunk_{k}" for k in range(len(chunks))],
            metadatas=[{"title": title} for _ in chunks]
        )
    print(f"  ✓ Indexed {len(docs)} video(s) into vector DB")
    return col


def vector_search(col, query, top_k=3):
    results = col.query(query_texts=[query], n_results=top_k)
    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        output.append((meta["title"], doc))
    return output


# ─── Ticker Extractor ────────────────────────────────────────────────────────

def extract_all_tickers(docs):
    """Pull every stock ticker mentioned across all videos"""
    # Match $TICKER or standalone ALL-CAPS 2-5 letter words (skip common words)
    skip = {"I", "A", "THE", "AND", "FOR", "BUT", "ARE", "NOT", "ALL", "YOU",
            "IT", "IN", "ON", "AT", "OR", "TO", "IS", "OF", "MY", "SO", "IF",
            "BE", "DO", "GO", "UP", "NOW", "NEW", "BIG", "GET", "HAS", "CAN",
            "MAY", "CEO", "ETF", "IPO", "GDP", "CPI", "FED", "SEC", "AI", "US",
            "SP", "QQQ", "SPY", "DCA", "IRA", "UK", "EU", "II", "III", "IV"}
    tickers = {}
    for title, text in docs:
        found = re.findall(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b', text)
        for match in found:
            t = match[0] or match[1]
            if t in skip: continue
            if t not in tickers:
                tickers[t] = {"count": 0, "videos": set()}
            tickers[t]["count"] += 1
            tickers[t]["videos"].add(title[:50])

    # Return sorted by frequency, filter noise (mentioned 2+ times)
    return sorted(
        [(t, d["count"], list(d["videos"])) for t, d in tickers.items() if d["count"] >= 2],
        key=lambda x: -x[1]
    )


# ─── Fallback Keyword Search ─────────────────────────────────────────────────

def keyword_search(docs, query, top_k=3):
    words = [w.lower() for w in query.split() if len(w) > 2]
    if not words:
        return []
    scored = []
    for title, text in docs:
        tl = text.lower()
        score = sum(tl.count(w) for w in words)
        if score > 0:
            pos = next((tl.find(w) for w in words if w in tl), 0)
            start = max(0, pos - 150)
            excerpt = ("..." if start > 0 else "") + text[start:start+600] + "..."
            scored.append((score, title, excerpt))
    return [(t, e) for _, t, e in sorted(scored, reverse=True)[:top_k]]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    channel = sys.argv[1] if len(sys.argv) > 1 else input("YouTube channel URL: ").strip()
    days    = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    docs = download_transcripts(channel, days)
    if not docs:
        print("\n❌ No transcripts found. Try more days: python3 poc.py <url> 60")
        return

    print(f"\n✅ Loaded {len(docs)} video(s)")

    # Vector DB (optional)
    collection = build_vector_index(docs)
    if collection:
        print("  ✓ Vector search active")
    else:
        print("  ℹ️  Keyword search  (pip install chromadb for vector search)")

    # AI (optional)
    try:
        import ollama
        use_ai = True
        print("  ✓ Ollama AI active  (llama3.2)")
    except ImportError:
        use_ai = False
        print("  ℹ️  No AI responses  (install Ollama for chat mode)")

    print("\nAsk anything. Special commands: 'tickers' = show all stocks mentioned.")
    print("─" * 60)

    while True:
        try:
            q = input("\nYou: ").strip()
            if not q:
                continue

            # Special command: extract all tickers
            if q.lower() in ("tickers", "stocks", "what stocks", "show tickers"):
                results = extract_all_tickers(docs)
                if not results:
                    print("No tickers found.")
                else:
                    print(f"\n📈 {len(results)} tickers mentioned:\n")
                    for ticker, count, videos in results[:30]:
                        print(f"  ${ticker:<6} ×{count:<3}  {videos[0][:55]}")
                print()
                continue

            if collection:
                results = vector_search(collection, q)
            else:
                results = keyword_search(docs, q)

            if not results:
                print("Nothing relevant found.")
                continue

            context = "\n---\n".join(f"[{t}]\n{e}" for t, e in results)

            if use_ai:
                r = ollama.chat("llama3.2", messages=[{
                    "role": "user",
                    "content": (
                        f"You are answering as a YouTube creator based ONLY on their transcripts.\n"
                        f"Question: {q}\n\nTranscripts:\n{context}\n\n"
                        f"Answer concisely, cite which video the info came from."
                    )
                }])
                print(f"\n🎙️  {r['message']['content']}")
            else:
                print(f"\nTop {len(results)} result(s):\n")
                for i, (title, excerpt) in enumerate(results, 1):
                    print(f"[{i}] {title}")
                    print(f"    {excerpt}\n")

        except KeyboardInterrupt:
            print("\n👋 Done!")
            break


if __name__ == "__main__":
    main()
