import re
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import gradio as gr
#  Extract YouTube Video ID from URL
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)
#  Fetch transcript from YouTube URL
def get_transcript_from_url(url):
    proxy_url = os.getenv("PROXY_URL")
    proxies = {
        "http": proxy_url,
        "https": proxy_url
    }
    video_id = extract_video_id(url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([entry['text'] for entry in transcript])
#  Summarize transcript safely
def summarize_transcript(text, max_chunk=500):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    summary = ""
    for chunk in chunks:
        word_count = len(chunk.strip().split())

        # ❗ Skip tiny chunks
        if word_count < 40:
            continue

        # 🔄 Dynamically adjust max/min length based on input size
        dynamic_max = min(130, int(word_count * 0.6))  # e.g. 60% of input length
        dynamic_min = max(10, int(dynamic_max * 0.5))   # min is 50% of max or at least 10

        try:
            result = summarizer(chunk, max_length=dynamic_max, min_length=dynamic_min, do_sample=False)
            summary += result[0]['summary_text'] + " "
        except Exception as e:
            print(f"⚠️ Skipping a chunk due to error: {e}")
            continue

    return summary.strip()

# ✅ Gradio interface function
def summarize_youtube_video(url):
    try:
        transcript = get_transcript_from_url(url)
        summary = summarize_transcript(transcript)
        return summary
    except Exception as e:
        return f"❌ Error: {e}"

# ✅ Launch Gradio UI
gr.Interface(
    fn=summarize_youtube_video,
    inputs=gr.Textbox(label="Enter YouTube Video URL"),
    outputs=gr.Textbox(label="Video Summary"),
    title="🎥 YouTube Transcript Summarizer",
    description="Paste any YouTube URL with subtitles. Summarizes the transcript using AI."
).launch()