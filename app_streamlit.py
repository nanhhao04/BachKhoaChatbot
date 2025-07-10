import streamlit as st
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from app.chatbot import get_chatbot_instance, StreamingCallbackHandler, check_data_availability
from app.data_processor import DataProcessor
from app.schema import Citation
import asyncio
import threading
import time

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="BachKhoa Support Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        align-items: flex-end;
    }
    .bot-message {
        background-color: #f5f5f5;
        align-items: flex-start;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .message-content {
        max-width: 80%;
    }
    .chat-history-item {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        cursor: pointer;
        border: 1px solid #e0e0e0;
    }
    .chat-history-item:hover {
        background-color: #f5f5f5;
    }
    .active-chat {
        background-color: #e3f2fd;
        border-color: #2196f3;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None

    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}

    if 'chatbot_ready' not in st.session_state:
        st.session_state.chatbot_ready = False

    if 'data_status' not in st.session_state:
        st.session_state.data_status = None

    if 'processing_query' not in st.session_state:
        st.session_state.processing_query = False


def check_chatbot_status():
    """Check if chatbot is ready and data is available"""
    try:
        data_status = check_data_availability()
        st.session_state.data_status = data_status

        if data_status.get('available', False):
            st.session_state.chatbot_ready = True
            return True
        else:
            st.session_state.chatbot_ready = False
            return False
    except Exception as e:
        logger.error(f"Error checking chatbot status: {e}")
        st.session_state.chatbot_ready = False
        st.session_state.data_status = {"available": False, "error": str(e)}
        return False


def create_new_chat():
    """Create a new chat session"""
    chat_id = f"chat_{int(time.time())}"
    st.session_state.current_chat_id = chat_id
    st.session_state.chat_sessions[chat_id] = {
        'messages': [],
        'created_at': datetime.now(),
        'title': f"Chat {len(st.session_state.chat_sessions) + 1}"
    }
    st.session_state.chat_history = []


def load_chat_session(chat_id: str):
    """Load a specific chat session"""
    if chat_id in st.session_state.chat_sessions:
        st.session_state.current_chat_id = chat_id
        st.session_state.chat_history = st.session_state.chat_sessions[chat_id]['messages']


def save_current_chat():
    """Save current chat to session"""
    if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chat_sessions:
        st.session_state.chat_sessions[st.session_state.current_chat_id]['messages'] = st.session_state.chat_history


def display_message(message: Dict[str, Any], is_user: bool = False):
    """Hiển thị tin nhắn trò chuyện, kèm trích dẫn nếu có"""
    css_class = "user-message" if is_user else "bot-message"
    sender = "👤 Bạn" if is_user else "🤖 Chatbot"

    # Hiển thị phần nội dung chính
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div class="message-header">{sender}</div>
        <div class="message-content">
            {message.get('content', '')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ✅ Hiển thị phần citations (nếu có)
    citations = message.get("citations", [])
    if citations:
        st.markdown(f"<div style='margin-top:10px; font-size: 90%;'><strong>🔗 Tham khảo:</strong></div>", unsafe_allow_html=True)

        for i, citation in enumerate(citations, start=1):
            text = citation.get("text", "")[:200] + "..."  # rút gọn
            source = citation.get("source", "#")

            st.markdown(f"""
            <div style='margin-bottom:10px; font-size: 90%; padding-left: 10px;'>
                <strong>Nguồn {i}:</strong> {text}<br>
                <a href="{source}" target="_blank">{source}</a>
            </div>
            """, unsafe_allow_html=True)


def display_chat_history():
    """Display chat history"""
    if not st.session_state.chat_history:
        st.info("💬 Chưa có tin nhắn nào. Hãy bắt đầu cuộc trò chuyện!")
        return

    for message in st.session_state.chat_history:
        display_message(message, message.get('is_user', False))


def process_user_query(query: str):
    """Process user query and get chatbot response"""
    if not query.strip():
        return

    # Add user message to history
    user_message = {
        'content': query,
        'is_user': True,
        'timestamp': datetime.now()
    }
    st.session_state.chat_history.append(user_message)

    try:
        st.session_state.processing_query = True

        # Get chatbot instance and process query
        chatbot = get_chatbot_instance()
        response = chatbot.answer_query(query)

        # Add bot response to history - Citations removed from storage
        bot_message = {
            'content': response.get('answer', 'Xin lỗi, tôi không thể trả lời câu hỏi này.'),
            'is_user': False,
            'timestamp': datetime.now()
        }
        st.session_state.chat_history.append(bot_message)

        # Save chat session
        save_current_chat()

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        error_message = {
            'content': f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}",
            'is_user': False,
            'timestamp': datetime.now()
        }
        st.session_state.chat_history.append(error_message)

    finally:
        st.session_state.processing_query = False


def sidebar():
    """Display sidebar with chat management and data management"""
    st.sidebar.title("🤖 BachKhoa Support Chatbot")

    # Data status section removed from sidebar

    # Chat management
    st.sidebar.header("💬 Quản lý Chat")

    if st.sidebar.button("➕ Chat mới", key="new_chat"):
        create_new_chat()
        st.rerun()

    # Chat history
    if st.session_state.chat_sessions:
        st.sidebar.subheader("📝 Lịch sử Chat")
        for chat_id, chat_data in st.session_state.chat_sessions.items():
            is_active = chat_id == st.session_state.current_chat_id
            button_text = f"{'📌' if is_active else '💬'} {chat_data['title']}"

            if st.sidebar.button(button_text, key=f"load_{chat_id}"):
                load_chat_session(chat_id)
                st.rerun()

    # Memory management
    st.sidebar.header("🧠 Quản lý bộ nhớ")

    if st.sidebar.button("🗑️ Xóa bộ nhớ", key="clear_memory"):
        try:
            if st.session_state.chatbot_ready:
                chatbot = get_chatbot_instance()
                chatbot.clear_memory()
            st.sidebar.success("Đã xóa bộ nhớ chatbot!")
        except Exception as e:
            st.sidebar.error(f"Lỗi khi xóa bộ nhớ: {e}")

    # Data management
    st.sidebar.header("📁 Quản lý dữ liệu")

    with st.sidebar.expander("🔧 Cài đặt nâng cao"):
        st.write("**Thông tin kết nối:**")
        st.write(f"- Milvus Host: {os.getenv('MILVUS_HOST', 'localhost')}")
        st.write(f"- Milvus Port: {os.getenv('MILVUS_PORT', '19530')}")
        st.write(f"- Collection: {os.getenv('MILVUS_COLLECTION', 'student_support_chatbot')}")

        if st.button("🔄 Tải lại ứng dụng", key="reload_app"):
            st.rerun()


def main():
    """Main application"""
    # Initialize session state
    init_session_state()

    # Check chatbot status on first load - but don't display status
    if st.session_state.data_status is None:
        check_chatbot_status()

    # Display sidebar
    sidebar()

    # Main content
    st.title("🎓 BachKhoa Support Chatbot")
    st.markdown("*Hệ thống hỗ trợ sinh viên Bách Khoa thông minh với RAG (Retrieval-Augmented Generation)*")

    # Data status display removed completely

    # Main chat interface
    if not st.session_state.chatbot_ready:
        st.error("⚠️ Chatbot chưa sẵn sàng. Vui lòng kiểm tra kết nối database và dữ liệu.")
        st.info("""
        **Hướng dẫn khắc phục:**
        1. Đảm bảo Milvus server đang chạy
        2. Kiểm tra dữ liệu đã được tải vào database chưa
        3. Chạy `python data_processor.py --pdf-dir data/pdfs --action create` để tạo dữ liệu
        """)
        return

    # Create new chat if none exists
    if not st.session_state.current_chat_id:
        create_new_chat()

    # Chat interface
    st.header("💬 Trò chuyện")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        display_chat_history()

    # Query input
    st.markdown("---")

    with st.form(key="query_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])

        with col1:
            query = st.text_input(
                "Nhập câu hỏi của bạn:",
                placeholder="Ví dụ: Điều kiện để được học bổng khuyến học là gì?",
                disabled=st.session_state.processing_query
            )

        with col2:
            submit_button = st.form_submit_button(
                "Gửi 📤",
                disabled=st.session_state.processing_query
            )

    # Process query
    if submit_button and query:
        with st.spinner("Đang xử lý câu hỏi..."):
            process_user_query(query)
            st.rerun()

    # Show processing status
    if st.session_state.processing_query:
        st.info("🤔 Đang suy nghĩ...")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Student Support Chatbot - Powered by RAG Technology<br>
        Built with Streamlit, LangChain, and Milvus
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()