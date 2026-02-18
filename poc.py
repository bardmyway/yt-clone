#!/usr/bin/env python3
"""
YT Clone - Minimal POC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Install:  pip install yt-dlp
Optional: pip install ollama  (then: ollama pull llama3.2)

Usage:
  python poc.py "https://www.youtube.com/@InvestingSimplified/videos" 14
  python poc.py "https://www.youtube.com/@mkbhd" 7

No OpenAI. No API keys. No vector DB. Works 100% offline.
"""

import subprocess, sys, os, re
from datetime import datetime, timedelta


def download_transcripts(channel_url, days=14):
    after = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    os.makedirs("transcripts", exist_ok=True)

    print(f"📥 Fetching last {days} days of transcripts...")
    subprocess.run([
        "yt-dlp",
        "--write-auto-subs", "--skip-download",
        "--sub-langs", "en",
        "--sub-format", "vtt",
        "--dateafter", after,
        "--playlist-end", "20",
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
                print(f"  ✓ {title[:65]}")

    return docs


def vtt_to_text(path):
    """Strip VTT timestamps, return clean text"""
    with open(path, encoding="utf-8", errors="ignore") as f:
        content = f.read()
    lines = []
    seen = set()
    for line in content.splitlines():
        line = line.strip()
        if not line: continue
        if re.match(r"^[\d:,\. ]+-->", line): continue   # timestamp
        if line in ("WEBVTT", "Kind: captions"): continue
        if line.startswith(("Language:", "NOTE", "align:", "position:")): continue
        if re.match(r"^\d+$", line): continue             # cue number
        # Deduplicate repeated captions
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return " ".join(lines)


def search(docs, query, top_k=3):
    """BM25-style keyword search — no ML required"""
    words = [w.lower() for w in query.split() if len(w) > 2]
    if not words:
        return []

    scored = []
    for title, text in docs:
        tl = text.lower()
        score = sum(tl.count(w) for w in words)
        if score > 0:
            # Find best excerpt around first keyword hit
            pos = next((tl.find(w) for w in words if w in tl), 0)
            start = max(0, pos - 150)
            excerpt = ("..." if start > 0 else "") + text[start:start + 600] + "..."
            scored.append((score, title, excerpt))

    return sorted(scored, reverse=True)[:top_k]


def main():
    channel = sys.argv[1] if len(sys.argv) > 1 else input("YouTube channel URL: ").strip()
    days    = int(sys.argv[2]) if len(sys.argv) > 2 else 14

    docs = download_transcripts(channel, days)

    if not docs:
        print("\n❌ No transcripts found. Try a larger date range or check the channel URL.")
        return

    print(f"\n✅ Loaded {len(docs)} video(s)\n")

    # Optional: Ollama for real AI responses
    try:
        import ollama
        use_ai = True
        print("🤖 Ollama detected — AI mode ON  (model: llama3.2)")
    except ImportError:
        use_ai = False
        print("🔍 Search mode — showing raw excerpts  (install Ollama for AI responses)")

    print("Ask anything. Ctrl+C to quit.\n")
    print("─" * 60)

    while True:
        try:
            q = input("\nYou: ").strip()
            if not q:
                continue

            results = search(docs, q)

            if not results:
                print("No relevant content found. Try different keywords.")
                continue

            context = "\n---\n".join(f"[{t}]\n{e}" for _, t, e in results)

            if use_ai:
                r = ollama.chat("llama3.2", messages=[{
                    "role": "user",
                    "content": (
                        f"You are answering as a YouTube creator based ONLY on their video transcripts below.\n"
                        f"Question: {q}\n\n"
                        f"Transcripts:\n{context}\n\n"
                        f"Answer in the creator's voice, citing which video the info came from."
                    )
                }])
                print(f"\n🎙️  {r['message']['content']}")
            else:
                print(f"\nTop {len(results)} match(es):\n")
                for i, (score, title, excerpt) in enumerate(results, 1):
                    print(f"[{i}] {title}")
                    print(f"    {excerpt}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Done!")
            break


if __name__ == "__main__":
    main()
