# YT Clone — Chat with Any YouTube Creator

Turn any YouTube channel into a searchable knowledge base. Ask questions, extract stock picks, find specific topics — all from the creator's actual video transcripts.

**No OpenAI. No API keys. 100% local.**

---

## What It Does

```
You: what stocks did he mention?

[1] The AI Power Bottleneck Just Got Deeper
    ...here's the seven stocks. First is APLD (Applied Digital Corp).
    They're the leader in building purpose-built AI and HPC data centers...

[2] What happens next week will shape stockmarket for the next DECADE
    ...Warren AI came up with Cotera Energy and Skyward Specialty Insurance
    with a 7.4% free cash flow yield and 3.4% dividend...
```

---

## Quick Start (Mac / Work Laptop)

### Step 1 — Install yt-dlp
```bash
brew install pipx && pipx install yt-dlp && pipx ensurepath
# Open a new terminal after this
```

### Step 2 — Set up Python environment

**Recommended (Python 3.12 — full semantic search):**
```bash
brew install python@3.12
python3.12 -m venv ~/yt-venv
source ~/yt-venv/bin/activate
pip install yt-dlp chromadb
```

**If you only have Python 3.14+ (TF-IDF search):**
```bash
python3 -m venv ~/yt-venv
source ~/yt-venv/bin/activate
pip install yt-dlp scikit-learn
```

### Step 3 — Download the script
```bash
curl -O https://raw.githubusercontent.com/bardmyway/yt-clone/main/poc.py
```
> ⚠️ If the repo is private, download `poc.py` via GitHub UI: open the file → click **Raw** → Save As

### Step 4 — Run
```bash
source ~/yt-venv/bin/activate   # if not already active
python3 poc.py "https://www.youtube.com/@NolanGouveia/videos" 30
```

---

## Usage

```bash
# Any YouTube channel, last N days
python3 poc.py "https://www.youtube.com/@NolanGouveia/videos" 30
python3 poc.py "https://www.youtube.com/@mkbhd" 14
python3 poc.py "https://www.youtube.com/@InvestingSimplified/videos" 60

# Single video
python3 poc.py "https://youtu.be/VIDEO_ID" 7
```

### Special Commands (type these at the prompt)
| Command | What it does |
|---------|-------------|
| `tickers` | Extract all stock tickers mentioned across all videos |
| `stocks` | Same as tickers |
| Any question | Search transcripts for relevant passages |

---

## Search Quality Tiers

| Setup | Quality | Install |
|-------|---------|---------|
| Keyword only | ⭐⭐⭐ | Just `yt-dlp` (zero setup) |
| TF-IDF | ⭐⭐⭐⭐ | `pip install scikit-learn` |
| Semantic (ChromaDB) | ⭐⭐⭐⭐⭐ | Python 3.12 + `pip install chromadb` |
| AI Chat mode | ⭐⭐⭐⭐⭐+ | Above + Ollama (see below) |

**ChromaDB** downloads `all-MiniLM-L6-v2` (~79MB, one-time) and uses real sentence embeddings — it understands that "what stocks did he recommend" and "best equity picks" mean the same thing.

---

## Add AI Responses (Optional)

Install [Ollama](https://ollama.ai), then:
```bash
ollama pull llama3.2
pip install ollama
```
Re-run `poc.py` — it auto-detects Ollama and switches to full chat mode where the AI answers in the creator's voice.

---

## Full Pipeline (Advanced / Home Setup)

For persistent vector DB, multi-channel indexing, and production use:

```bash
pip install -r requirements.txt

# 1. Download transcripts
python3 yt_transcripts.py "https://www.youtube.com/@NolanGouveia/videos" 90

# 2. Build vector index
python3 build_index.py

# 3. Chat
python3 chat.py
```

---

## How It Works

1. `yt-dlp` downloads auto-generated subtitles (VTT format) — no video download needed
2. VTT timestamps are stripped → clean transcript text
3. Transcripts are chunked into 400-char overlapping windows
4. ChromaDB embeds each chunk with `all-MiniLM-L6-v2` (local, offline)
5. Your query is embedded the same way → cosine similarity finds the best matches
6. Optionally: Ollama generates a natural language answer from the matched chunks

---

## Example Channels to Try

- `https://www.youtube.com/@NolanGouveia/videos` — investing / stock picks
- `https://www.youtube.com/@mkbhd` — tech reviews
- `https://www.youtube.com/@lexfridman` — AI / science interviews
- `https://www.youtube.com/@InvestingSimplified/videos` — dividend investing

---

## Requirements

- Python 3.10+
- `yt-dlp` (via pipx or pip)
- `chromadb` (optional, recommended — Python 3.12)
- `scikit-learn` (optional fallback — Python 3.14+)
- `ollama` (optional — for AI chat responses)
