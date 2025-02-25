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
COMPETITOR_NAMES = list(COMPETITORS.values())

# Vector Store Setup
embedding_function = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=OPENAI_API_KEY)
vectorstore = Chroma(persist_directory="data/podcast_chroma.db", embedding_function=embedding_function)
if not os.path.exists("data/podcast_chroma.db"):
    st.error("Vector store not found. Run the data collection script.")
    st.stop()

# Enhanced Database Summary with Full Context
def generate_db_summary(df, vectorstore):
    podcast_counts = df['podcast_name'].value_counts().to_dict()
    metadata_fields = list(df.columns)
    total_episodes = len(df)
    sample_docs = vectorstore.similarity_search("common topics", k=5)
    sample_text = " ".join([doc.page_content[:200] for doc in sample_docs])
    avg_views = df.groupby('podcast_name')['view_count'].mean().round().astype(int).to_dict()
    bartlett_df = df[df['podcast_name'] == "Steven Bartlett – Diary of a CEO"] \
                    .sort_values('view_count', ascending=False).head(15).to_string(index=False)
    summary = (
        f"Database Overview:\n"
        f"- Podcasts: {', '.join(COMPETITOR_NAMES)}\n"
        f"- Episode Counts: {', '.join([f'{name}: {count}' for name, count in podcast_counts.items()])}\n"
        f"- Avg Views: {', '.join([f'{name}: {views}' for name, views in avg_views.items()])}\n"
        f"- Metadata Fields: {', '.join(metadata_fields)}\n"
        f"- Total Episodes: {total_episodes}\n"
        f"- Transcript Sample: Common topics include {sample_text[:100]}...\n"
        f"- Sample Data for Steven Bartlett – Diary of a CEO (top 15 by view_count):\n{bartlett_df}\n"
        f"Creators map to podcasts: {', '.join([f'{k} → {v}' for k, v in COMPETITORS.items()])}.\n"
        f"Full metadata is in 'df' with {len(df)} rows—use ALL available data for queries.\n\n"
        "{context}"  # This placeholder will be replaced by retrieved docs
    )
    return summary

db_summary = generate_db_summary(df, vectorstore)

# Chat History Setup
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if "welcome_added" not in st.session_state:
    st.session_state.welcome_added = False
if not msgs.messages and not st.session_state.welcome_added:
    msgs.add_ai_message(f"I have data on these podcasts: {', '.join(COMPETITOR_NAMES)}. Ask me anything—I’ll use all the data to answer precisely!")
    st.session_state.welcome_added = True

# LLM-Driven RAG Chain with Fixed Input
def create_rag_chain(df):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=OPENAI_API_KEY, max_tokens=4000)

    # Contextualization prompt for the retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Reformulate the question as a standalone query based on chat history, using creator names (e.g., 'Jay Shetty') to infer the podcast if clear."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    contextualize_q_prompt.input_variables = ["input", "chat_history"]

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_system_prompt = (
        f"{db_summary}\n"
        "You are a highly intelligent assistant analyzing a podcast database. Answer naturally and accurately using:\n"
        "- Transcripts: Context below for content insights.\n"
        "- Metadata: 'df' (columns: {', '.join(df.columns)}) for stats. Treat 'df' as a database—filter, sort, group freely.\n"
        "Critical Rules (MUST FOLLOW WITHOUT EXCEPTION):\n"
        "- For 'top N' requests (e.g., 'top 10 videos'), YOU MUST RETURN EXACTLY N ITEMS IF THEY EXIST IN 'df'. IF FEWER THAN N EXIST, EXPLAIN: 'I could only find X entries for Y in df.'\n"
        "- Map creators (e.g., 'Steven Bartlett') to podcasts via COMPETITORS.\n"
        "- Metrics: Infer from context (e.g., 'videos' → view_count, 'likes' → like_count) or default to view_count.\n"
        "- Format: Numbered lists for top items (e.g., '1. Title (date): X views'). Blend stats and content for insights.\n"
        "- Debugging: If asked for N items but fewer are returned, explain why.\n"
        "Expect input as a dictionary with an 'input' key (e.g., {'input': 'Top 10 videos for Steven Bartlett'}). Use ALL data in 'df'. If data is missing, say 'I don’t have that info'. Use chat history for context.\n\n"
        "{context}"  # Placeholder for retrieved documents
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_prompt.input_variables = ["input", "chat_history", "context"]

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

    if question := st.chat_input("E.g., 'Top 10 videos for Steven Bartlett' or 'Why is Jay Shetty popular?'"):
        with st.spinner("Analyzing..."):
            msgs.add_user_message(question)
            if show_history:
                st.chat_message("human").write(question)

            # Supply default empty strings for missing keys
            user_input = {"input": question, "chat_history": "", "context": ""}
            st.write("Input to rag_chain:", user_input)
            response = rag_chain.invoke(user_input, config={"configurable": {"session_id": "any"}})
            st.write("Response from rag_chain:", response)
            answer = response['answer']

            if "steven bartlett" in question.lower():
                bartlett_count = len(df[df['podcast_name'] == "Steven Bartlett – Diary of a CEO"])
                answer += f"\n\n[Debug: {bartlett_count} entries available for Steven Bartlett – Diary of a CEO in df]"

            msgs.add_ai_message(answer)
            if show_history:
                st.chat_message("ai").write(answer)

# Tab 3: Content Trends
with tab3:
    st.subheader("Trending Topics & Insights")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Topic Summary"):
            with st.spinner("Extracting trends..."):
                user_input = {"input": "Summarize the most common topics across all podcasts with examples.", "chat_history": "", "context": ""}
                response = rag_chain.invoke(user_input, config={"configurable": {"session_id": "any"}})
                st.write(response['answer'])
    with col2:
        if st.button("Predict Next Big Topic"):
            with st.spinner("Predicting..."):
                user_input = {"input": "Based on trends and content, predict the next big topic for these podcasts.", "chat_history": "", "context": ""}
                response = rag_chain.invoke(user_input, config={"configurable": {"session_id": "any"}})
                st.write(response['answer'])
