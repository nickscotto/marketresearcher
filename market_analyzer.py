import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import os
import streamlit as st
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import plotly.express as px
from datetime import datetime

# Streamlit Config
st.set_page_config(page_title="Podcast Competitor Analyzer", layout="wide")
st.title("Podcast Competitor Analyzer")

# Load API Keys from Streamlit Secrets
try:
    YOUTUBE_API_KEY = st.secrets["youtube"]
    OPENAI_API_KEY = st.secrets["openai"]
except KeyError:
    st.error("API keys for YouTube and OpenAI are missing in Streamlit secrets.")
    st.stop()

# Load Metadata with Diagnostics
if os.path.exists('competitor_podcast_videos.csv'):
    df = pd.read_csv('competitor_podcast_videos.csv')
    df['published_at'] = pd.to_datetime(df['published_at'])
    # Diagnostic: Check rows per podcast
    st.write("Data Loaded - Row Counts per Podcast:")
    st.write(df['podcast_name'].value_counts())
else:
    st.error("Run the data collection script first to generate 'competitor_podcast_videos.csv'.")
    st.stop()

# Competitors List with Creator Mapping
COMPETITORS = {
    "Russell Brand": "Russell Brand – Stay Free",
    "Dr. Rangan Chatterjee": "Dr. Rangan Chatterjee – Feel Better, Live More",
    "Jay Shetty": "Jay Shetty – On Purpose",
    "Tom Bilyeu": "Tom Bilyeu – Impact Theory",
    "Lewis Howes": "Lewis Howes – The School of Greatness",
    "Vishen Lakhiani": "Mindvalley – Vishen Lakhiani",
    "Steven Bartlett": "Steven Bartlett – Diary of a CEO"
}
COMPET
