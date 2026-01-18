import streamlit as st
from googleapiclient.discovery import build
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np
import re
from dotenv import load_dotenv
import os
import pandas as pd

# === Your YouTube Data API key ===
load_dotenv()
YOUTUBE_API_KEY = st.secrets.get("API_KEY") or os.getenv("API_KEY")
if not YOUTUBE_API_KEY:
    st.error("Missing API_KEY. Add it in Streamlit Cloud → Manage app → Secrets.")
    st.stop()

# Build YouTube client
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY, cache_discovery=False)

# === Load your fine-tuned model ===
@st.cache_resource
def load_model_and_tokenizer():
    model = TFAutoModelForSequenceClassification.from_pretrained("roberta_web_model")
    tokenizer = AutoTokenizer.from_pretrained("roberta_web_model")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# === Utility: Extract video ID from URL ===
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

# === Fetch comments from YouTube API ===
def fetch_comments(video_id, max_comments=1000):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )
    response = request.execute()
    while response and len(comments) < max_comments:
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_comments:
                break
        request = youtube.commentThreads().list_next(request, response)
        response = request.execute() if request else None
    return comments

# === Run inference ===
def predict_sentiment(comments):
    inputs = tokenizer(comments, return_tensors="tf", padding=True, truncation=True, max_length=128)
    dataset = tf.data.Dataset.from_tensor_slices(inputs).batch(32)

    predictions = []
    for batch in dataset:
        outputs = model(batch)
        probs = tf.nn.softmax(outputs.logits, axis=1).numpy()
        predictions.append(probs)

    probs = np.concatenate(predictions, axis=0)
    preds = np.argmax(probs, axis=1)
    labels = ['negative', 'neutral', 'positive']
    results = [
        {"comment": comment, "sentiment": labels[preds[i]], "confidence": float(probs[i][preds[i]])}
        for i, comment in enumerate(comments)
    ]
    return results

# === Streamlit UI ===
st.title("🎥 YouTube Comment Sentiment Analyzer")

url = st.text_input("Enter a YouTube video URL:")

if st.button("Fetch and Analyze Comments"):
    video_id = extract_video_id(url)
    if not video_id:
        st.error("❌ Invalid YouTube URL")
    else:
        with st.spinner("Fetching comments..."):
            comments = fetch_comments(video_id)
        if not comments:
            st.warning("No comments found.")
        else:
            st.success(f"Fetched {len(comments)} comments.")
            with st.spinner("Analyzing sentiment..."):
                results = predict_sentiment(comments)

            st.write("### 💡 Results:")

            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            # Small summary
            st.write(df["sentiment"].value_counts(normalize=True).rename("share").mul(100).round(1))

            # Showing every comments individually is too slow
            # for r in results:
            #     st.markdown(f"💬 **{r['comment']}**")
            #     st.write(f"→ Sentiment: **{r['sentiment']}** (Confidence: {r['confidence']:.2f})")
            #     st.markdown("---")
                