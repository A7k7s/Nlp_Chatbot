import streamlit as st
import time
from datetime import datetime
from chatbot import Chatbot

# Page configuration
st.set_page_config(
    page_title="EduBot - College Assistant",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stSidebar {
        background-color: #f8f9fa;
    }
    .sentiment-tag {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: -10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize chatbot
@st.cache_resource
def load_bot():
    return Chatbot("data.json")

bot = load_bot()

# Sidebar
with st.sidebar:
    st.title("🎓 EduBot")
    st.markdown("---")
    st.markdown("### About Project")
    st.info("""
    **EduBot** is a lightweight NLP-based chatbot designed for low-resource machines.
    
    **Features:**
    - TF-IDF + Cosine Similarity matching
    - spaCy preprocessing
    - Sentiment Detection
    - Typing Effect
    """)
    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main UI
st.title("Chat with EduBot 🤖")
st.caption("Your campus companion for admissions, fees, and more!")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(f"{message['timestamp']} | Sentiment: {message['sentiment']}")

# Chat input
if prompt := st.chat_input("Ask me about admissions, fees, or courses..."):
    # Display user message
    timestamp = datetime.now().strftime("%I:%M %p")
    
    # Get response from chatbot
    response, sentiment = bot.get_response(prompt)
    
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp,
        "sentiment": sentiment
    })
    
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"{timestamp} | Sentiment: {sentiment}")

    # Display bot response with typing effect
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate typing effect
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        bot_timestamp = datetime.now().strftime("%I:%M %p")
        st.caption(f"{bot_timestamp} | Sentiment: Neutral")
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": bot_timestamp,
            "sentiment": "Neutral"
        })
