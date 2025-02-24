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
import re
from textblob import TextBlob  # For basic sentiment analysis
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

# Load Metadata
if os.path.exists('competitor_podcast_videos.csv'):
    df = pd.read_csv('competitor_podcast_videos.csv')
    df['published_at'] = pd.to_datetime(df['published_at'])  # Ensure date compatibility
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
COMPETITOR_NAMES = list(COMPETITORS.values())

# Vector Store Setup
embedding_function = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=OPENAI_API_KEY)
vectorstore = Chroma(persist_directory="data/podcast_chroma.db", embedding_function=embedding_function)
if not os.path.exists("data/podcast_chroma.db"):
    st.error("Vector store not found. Run the data collection script.")
    st.stop()

# Enhanced Database Summary
def generate_db_summary(df, vectorstore):
    podcast_counts = df['podcast_name'].value_counts().to_dict()
    metadata_fields = list(df.columns)
    total_episodes = len(df)
    sample_docs = vectorstore.similarity_search("common topics", k=5)
    sample_text = " ".join([doc.page_content[:200] for doc in sample_docs])
    avg_views = df.groupby('podcast_name')['view_count'].mean().round().astype(int).to_dict()
    summary = (
        f"Database Overview:\n"
        f"- Podcasts: {', '.join(COMPETITOR_NAMES)}\n"
        f"- Episode Counts: {', '.join([f'{name}: {count}' for name, count in podcast_counts.items()])}\n"
        f"- Avg Views: {', '.join([f'{name}: {views}' for name, views in avg_views.items()])}\n"
        f"- Metadata Fields: {', '.join(metadata_fields)} (e.g., view_count, like_count, published_at)\n"
        f"- Total Episodes: {total_episodes}\n"
        f"- Transcript Sample: Common topics include {sample_text[:100]}...\n"
        f"Creators map to podcasts: {', '.join([f'{k} → {v}' for k, v in COMPETITORS.items()])}."
    )
    return summary

db_summary = generate_db_summary(df, vectorstore)

# Chat History Setup
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if "welcome_added" not in st.session_state:
    st.session_state.welcome_added = False
if not msgs.messages and not st.session_state.welcome_added:
    msgs.add_ai_message(f"I have data on these podcasts: {', '.join(COMPETITOR_NAMES)}. Ask me anything—top videos, trends, comparisons, or deep dives!")
    st.session_state.welcome_added = True

# Advanced Stats and Analysis Functions
def fetch_advanced_stats(query_type, podcast_names, n=5, metric="view_count", time_period=None):
    if not all(p in COMPETITOR_NAMES for p in podcast_names):
        invalid = [p for p in podcast_names if p not in COMPETITOR_NAMES]
        return f"I don’t have data on {', '.join(invalid)}. I only track: {', '.join(COMPETITOR_NAMES)}."
    
    filtered_df = df[df['podcast_name'].isin(podcast_names)]
    if time_period:
        today = datetime.now()
        if "last year" in time_period.lower():
            filtered_df = filtered_df[filtered_df['published_at'] > today.replace(year=today.year - 1)]
    
    if filtered_df.empty:
        return f"No data found for {', '.join(podcast_names)} in the specified scope."

    if query_type == "top_videos":
        result = f"Top {n} videos for {', '.join(podcast_names)} by {metric}:\n"
        top_df = filtered_df.sort_values(metric, ascending=False).head(n)
        for i, row in top_df.iterrows():
            result += f"- '{row['title']}' ({row['podcast_name']}, {row['published_at'].date()}): {row[metric]} {metric.replace('_count', 's')}\n"
        return result
    
    elif query_type == "comparison":
        stats = filtered_df.groupby('podcast_name').agg({
            'view_count': ['mean', 'sum'],
            'like_count': ['mean'],
            'comment_count': ['mean']
        }).round().astype(int)
        result = f"Comparison of {', '.join(podcast_names)}:\n"
        for podcast in podcast_names:
            if podcast in stats.index:
                result += (f"- {podcast}: Avg Views: {stats.loc[podcast, ('view_count', 'mean')]}, "
                          f"Total Views: {stats.loc[podcast, ('view_count', 'sum')]}, "
                          f"Avg Likes: {stats.loc[podcast, ('like_count', 'mean')]}, "
                          f"Avg Comments: {stats.loc[podcast, ('comment_count', 'mean')]}\n")
        return result
    
    elif query_type == "trend":
        trend_df = filtered_df.groupby([pd.Grouper(key='published_at', freq='M'), 'podcast_name'])[metric].mean().unstack().fillna(0)
        result = f"Trend analysis for {metric} over time:\n"
        for podcast in podcast_names:
            if podcast in trend_df.columns:
                recent = trend_df[podcast].iloc[-3:].mean().round()
                result += f"- {podcast}: Avg {metric} (last 3 months): {recent}\n"
        return result

# RAG Chain with Advanced Capabilities
def create_rag_chain(df):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=OPENAI_API_KEY, max_tokens=4000)  # Upgrade to gpt-4o for deeper analysis

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Reformulate the question as a standalone query based on chat history, using creator names (e.g., 'Jay Shetty') to infer the podcast if clear."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_system_prompt = (
        f"{db_summary}\n\n"
        "You are an advanced assistant analyzing a podcast database. Handle all queries with depth:\n"
        "- Content: Use transcripts in the context below for topics, strategies, or sentiment.\n"
        "- Stats: Return 'STATS:<query_type>:<podcast_names>:<n>:<metric>:<time_period>' (e.g., 'STATS:top_videos:Jay Shetty – On Purpose:10:view_count:last year'). "
        "  - query_type: 'top_videos', 'comparison', 'trend'.\n"
        "  - podcast_names: comma-separated list (e.g., 'Jay Shetty – On Purpose,Tom Bilyeu – Impact Theory').\n"
        "  - n: number of items (default 5).\n"
        "  - metric: 'view_count', 'like_count', 'comment_count' (default 'view_count').\n"
        "  - time_period: 'last year', 'all time' (default 'all time').\n"
        "Resolve creator names to podcasts via COMPETITORS. For complex queries (e.g., 'why is Steven Bartlett popular?'), combine stats and content insights. "
        "If data is missing, say 'I don’t have that info' and suggest the Dashboard. Use chat history for context.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

rag_chain = create_rag_chain(df)

# UI Layout: Tabs
tab1, tab2, tab3 = st.tabs(["Dashboard", "Chat Analyzer", "Content Trends"])

# Tab 1: Dashboard
with tab1:
    st.subheader("Competitor Stats")
    podcast_filter = st.multiselect("Select Podcasts", df['podcast_name'].unique(), default=df['podcast_name'].unique())
    filtered_df = df[df['podcast_name'].isin(podcast_filter)]
    col1, col2 = st.columns(2)
    with col1:
        st.write("Average Engagement")
        avg_stats = filtered_df.groupby('podcast_name')[['view_count', 'like_count', 'comment_count']].mean().reset_index()
        st.dataframe(avg_stats.style.format({"view_count": "{:.0f}", "like_count": "{:.0f}", "comment_count": "{:.0f}"}))
    with col2:
        st.write("Views Over Time")
        fig = px.line(filtered_df, x='published_at', y='view_count', color='podcast_name', title="Views by Episode")
        st.plotly_chart(fig)

# Tab 2: Chat Analyzer
with tab2:
    st.subheader("Ask About Your Competitors")
    show_history = st.checkbox("Show Conversation History", value=True)
    
    if show_history:
        for msg in msgs.messages:
            st.chat_message(msg.type).write(msg.content)
    else:
        st.write("History hidden. Toggle 'Show Conversation History' to view past messages.")

    if question := st.chat_input("E.g., 'Compare Jay Shetty and Steven Bartlett' or 'Why is Tom Bilyeu popular?'"):
        with st.spinner("Digging deep..."):
            msgs.add_user_message(question)
            if show_history:
                st.chat_message("human").write(question)

            response = rag_chain.invoke({"input": question}, config={"configurable": {"session_id": "any"}})
            response = response['answer']

            if response.startswith("STATS:"):
                parts = response.split(":")
                query_type = parts[1]
                podcast_names = parts[2].split(",")
                n = int(parts[3]) if parts[3] else 5
                metric = parts[4] if parts[4] else "view_count"
                time_period = parts[5] if len(parts) > 5 else None
                response = fetch_advanced_stats(query_type, podcast_names, n, metric, time_period)
            
            msgs.add_ai_message(response)
            if show_history:
                st.chat_message("ai").write(response)

# Tab 3: Content Trends
with tab3:
    st.subheader("Trending Topics & Insights")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Topic Summary"):
            with st.spinner("Extracting trends..."):
                response = rag_chain.invoke({"input": "Summarize the most common topics across all podcasts with examples."}, config={"configurable": {"session_id": "any"}})
                st.write(response['answer'])
    with col2:
        if st.button("Predict Next Big Topic"):
            with st.spinner("Predicting..."):
                response = rag_chain.invoke({"input": "Based on trends and content, predict the next big topic for these podcasts."}, config={"configurable": {"session_id": "any"}})
                st.write(response['answer'])