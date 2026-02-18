# YouTube Creator Clone

Create an AI clone of any YouTube creator by feeding their transcripts into a RAG pipeline.

## What It Does

1. **Downloads transcripts** from any YouTube channel for the last N days
2. **Builds a searchable knowledge base** using vector embeddings
3. **Lets you chat** with an AI that answers in the creator's style, using only their actual content

Perfect for:
- Cloning stock analysts to extract their strategies
- Learning from educational creators without watching hours of video
- Creating interactive "experts" you can query

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download transcripts from a channel
python yt_transcripts.py "https://www.youtube.com/@InvestingSimplified/videos" 30

# 3. Build the vector index
python build_index.py

# 4. Chat with the clone
python chat.py
```

## How It Works

### Step 1: Transcript Download (`yt_transcripts.py`)
Uses `yt-dlp` to download VTT subtitle files, then parses them to clean text. Handles:
- Auto-generated and manual captions
- Date filtering (last N days)
- Cookie extraction from your browser (to avoid blocks)

### Step 2: Index Building (`build_index.py`)
- Chunks transcripts into logical segments
- Embeds using `nomic-embed-text` (local, no API key)
- Stores in ChromaDB vector database
- Tags each chunk with metadata (creator, video, date)

### Step 3: Chat Interface (`chat.py`)
- Vector search for relevant content
- System prompt that mimics the creator's voice
- Context injection from actual transcripts
- Local LLM via Ollama (default: `llama3.2`) or OpenAI API

## Example: Stock Analyst Clone

```bash
# Download a year of "Investing Simplified - Professor G"
python yt_transcripts.py "https://www.youtube.com/@InvestingSimplified/videos" 365

# Build index
python build_index.py

# Chat
python chat.py

You: "What's your current thesis on energy stocks?"
Clone: "Based on my recent videos, I'm bullish on nuclear energy companies like..."
```

## Requirements

- Python 3.10+
- `yt-dlp` (installed via requirements.txt)
- Optional: Ollama for local LLM, or OpenAI API key

## Why This Works When Others Fail

YouTube blocks most automated transcript access. This pipeline uses:
1. `yt-dlp` with `--cookies-from-browser` to mimic real user access
2. Local embeddings to avoid API limits
3. No headless browser needed (faster, more reliable than Selenium)

## Project Structure

```
.
├── yt_transcripts.py      # Download and parse transcripts
├── build_index.py         # Create vector database
├── chat.py               # Query interface
├── requirements.txt      # Dependencies
├── transcripts/          # Downloaded transcripts (.txt)
├── chroma_db/           # Vector database
└── README.md
```

## License

MIT