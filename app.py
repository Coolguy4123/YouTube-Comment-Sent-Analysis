import streamlit as st
from googleapiclient.discovery import build
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np
import re
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
API_key = os.getenv("API_KEY")

def load_model():
    model = TFAutoModelForSequenceClassification.from_pretrained("roberta_web_model")
    tokenizer = AutoTokenizer.from_pretrained("roberta_web_model")
    return model, tokenizer

def extract_videoID(url):
    pattern = r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None
    
# -- Streamlit UI --
st.title("Youtube Comment Sentiment Analyzer")
url = st.text_input("Enter YouTube Video URL:")

