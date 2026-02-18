#!/usr/bin/env python3
"""
YouTube Channel Transcript Downloader
Downloads transcripts for all videos from a channel in the last N days.
Usage: python3 yt_transcripts.py <channel_url> [days=30]
"""

import subprocess
import sys
import os
import re
import json
from pathlib import Path
from datetime import datetime, timedelta

CHANNEL_URL = sys.argv[1] if len(sys.argv) > 1 else None
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
OUTPUT_DIR = Path("transcripts")
OUTPUT_DIR.mkdir(exist_ok=True)

if not CHANNEL_URL:
    print("Usage: python3 yt_transcripts.py <channel_url> [days=30]")
    print("Example: python3 yt_transcripts.py 'https://www.youtube.com/@InvestingSimplified/videos' 365")
    sys.exit(1)

dateafter = (datetime.now() - timedelta(days=DAYS)).strftime("%Y%m%d")

print(f"📡 Fetching transcripts from: {CHANNEL_URL}")
print(f"📅 Date range: last {DAYS} days (since {dateafter})")
print(f"📁 Output: {OUTPUT_DIR.resolve()}\n")

# Step 1: Download VTT subtitle files
cmd = [
    "yt-dlp",
    "--write-auto-subs",
    "--sub-lang", "en",
    "--sub-format", "vtt",
    "--skip-download",
    "--dateafter", dateafter,
    "--output", str(OUTPUT_DIR / "%(upload_date)s_%(id)s_%(title).60s.%(ext)s"),
    "--no-warnings",
    "--quiet",
    "--progress",
    CHANNEL_URL
]

# Try with cookies from browser first (more reliable)
cmd_cookies = cmd.copy()
cmd_cookies.insert(1, "--cookies-from-browser")
cmd_cookies.insert(2, "chrome")

print("⬇️  Downloading subtitle files (with browser cookies)...")
try:
    result = subprocess.run(cmd_cookies, capture_output=False, check=False)
except Exception as e:
    print(f"⚠️  Cookie method failed, trying without cookies: {e}")
    result = subprocess.run(cmd, capture_output=False, check=False)

# Step 2: Parse VTT → clean text
vtt_files = sorted(OUTPUT_DIR.glob("*.vtt"))
print(f"\n✅ Found {len(vtt_files)} subtitle files. Parsing...\n")

def parse_vtt(vtt_path):
    """Convert VTT to clean deduplicated text."""
    text = vtt_path.read_text(encoding="utf-8", errors="ignore")
    # Remove header
    text = re.sub(r"WEBVTT.*?\n\n", "", text, flags=re.DOTALL)
    # Remove timestamps and cue IDs
    lines = text.split("\n")
    clean = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+$", line):  # cue number
            continue
        if re.match(r"[\d:.]+ --> [\d:.]+", line):  # timestamp
            continue
        if line.startswith("<") and line.endswith(">"):  # tags
            continue
        # Remove inline tags like <00:00:01.000><c>word</c>
        line = re.sub(r"<[^>]+>", "", line)
        line = line.strip()
        if line:
            clean.append(line)
    
    # Deduplicate consecutive repeated lines (VTT has rolling captions)
    deduped = []
    prev = None
    for line in clean:
        if line != prev:
            deduped.append(line)
            prev = line
    
    return " ".join(deduped)

# Process each VTT and save as .txt
results = []
for vtt in vtt_files:
    txt_path = vtt.with_suffix(".txt")
    transcript = parse_vtt(vtt)
    txt_path.write_text(transcript, encoding="utf-8")
    
    # Extract metadata from filename
    filename = vtt.stem
    parts = filename.split("_", 2)
    if len(parts) >= 3:
        upload_date, video_id, title = parts[0], parts[1], parts[2]
    else:
        upload_date, video_id, title = "unknown", "unknown", filename
    
    word_count = len(transcript.split())
    print(f"  ✓ {upload_date} — {title[:50]}... — {word_count:,} words")
    
    # Save metadata
    meta = {
        "upload_date": upload_date,
        "video_id": video_id,
        "title": title,
        "word_count": word_count,
        "transcript_file": str(txt_path.name)
    }
    meta_path = vtt.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
    results.append(meta)

print(f"\n🎉 Done! {len(results)} transcripts saved to ./{OUTPUT_DIR}/")
print(f"📊 Total words: {sum(r['word_count'] for r in results):,}")

# Save summary
summary = {
    "channel_url": CHANNEL_URL,
    "days": DAYS,
    "date_ran": datetime.now().isoformat(),
    "total_transcripts": len(results),
    "total_words": sum(r['word_count'] for r in results),
    "transcripts": results
}
summary_path = OUTPUT_DIR / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"📋 Summary saved to: {summary_path}")