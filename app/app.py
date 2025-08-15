import streamlit as st
import time
import uuid

from assistant import get_answer
from db import (
    save_conversation,
    save_feedback,
    get_recent_conversations,
    get_feedback_stats,
)


def print_log(message):
    print(message, flush=True)


OPENAI_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-3.5-turbo",
]


def init_states():
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
        print_log(f"New conversation started with ID: {st.session_state.conversation_id}")

    # 最近一次成功保存到数据库的会话ID，用于反馈关联
    if "last_conversation_id" not in st.session_state:
        st.session_state.last_conversation_id = None

    # 仅做页面内展示的计数器
    if "count" not in st.session_state:
        st.session_state.count = 0
        print_log("Feedback count initialized to 0")


def main():
    print_log("Starting the Course Assistant application")
    st.set_page_config(page_title="Course Assistant", page_icon="🎓")
    st.title("🎓 Course Q&A Assistant")

    init_states()

    # ======== 侧边栏设置 ========
    with st.sidebar:
        st.header("Settings")

        course = st.selectbox(
            "Course",
            ["machine-learning-zoomcamp", "data-engineering-zoomcamp", "mlops-zoomcamp"],
            index=0,
        )
        model_choice = st.selectbox(
            "OpenAI Model",
            OPENAI_MODELS,
            index=0,
            help="Only OpenAI models are used",
        )
        search_limit = st.slider(
            "Search top-K documents",
            min_value=1,
            max_value=10,
            value=5,
            help="The number of documents returned by the RAG search",
        )
        evaluate = st.checkbox(
            "Run relevance evaluation",
            value=True,
            help="Use an evaluation prompt to have the LLM evaluate the relevance of the answer to the question.",
        )

        st.markdown("---")
        st.caption(f"Conversation ID: `{st.session_state.conversation_id}`")
        if st.button("🔄 New conversation ID"):
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.last_conversation_id = None
            st.info("Started a new conversation ID for the next question.")

    # ======== 主区：提问与回答 ========
    with st.form("qa_form", clear_on_submit=False):
        user_input = st.text_input("Ask a question:", placeholder="Type your course-related question here...")
        submitted = st.form_submit_button("Ask")

    if submitted:
        if not user_input.strip():
            st.warning("Please enter a question before asking.")
        else:
            print_log(f"User asked: '{user_input}'")
            with st.spinner("Thinking..."):
                print_log(f"Getting answer using {model_choice} (topK={search_limit}, evaluate={evaluate})")
                t0 = time.time()
                answer_data = get_answer(
                    query=user_input,
                    course=course,
                    model_choice=model_choice,
                    search_limit=search_limit,
                    evaluate=evaluate,
                )
                t1 = time.time()
                print_log(f"Answer received in {t1 - t0:.2f} seconds")

            # 展示答案与指标
            st.success("Completed!")
            st.markdown("**Answer:**")
            st.write(answer_data["answer"])

            cols = st.columns(4)
            cols[0].metric("Response time (s)", f"{answer_data['response_time']:.2f}")
            cols[1].metric("Model", answer_data["model_used"])
            cols[2].metric("Total tokens", answer_data["total_tokens"])
            if answer_data.get("openai_cost", 0) > 0:
                cols[3].metric("OpenAI cost (USD)", f"{answer_data['openai_cost']:.4f}")
            else:
                cols[3].metric("OpenAI cost (USD)", "—")

            if evaluate:
                st.info(
                    f"Relevance: **{answer_data.get('relevance', 'N/A')}** — {answer_data.get('relevance_explanation', '')}"
                )

            # 保存到数据库
            try:
                print_log("Saving conversation to database")
                save_conversation(st.session_state.conversation_id, user_input, answer_data, course)
                st.session_state.last_conversation_id = st.session_state.conversation_id
                print_log("Conversation saved successfully")

                # 生成新ID供下一次提问使用
                st.session_state.conversation_id = str(uuid.uuid4())
            except Exception as e:
                st.error(f"Failed to save conversation: {e}")

    # ======== 反馈区域（绑定到最后一次成功保存的会话） ========
    st.subheader("Feedback")
    disabled_feedback = st.session_state.last_conversation_id is None
    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])

    with fb_col1:
        if st.button("👍 +1", disabled=disabled_feedback, help="Positive feedback for the last answer"):
            try:
                st.session_state.count += 1
                save_feedback(st.session_state.last_conversation_id, 1)
                st.success("Thanks for your feedback (+1)!")
                print_log("Positive feedback saved to database")
            except Exception as e:
                st.error(f"Failed to save feedback: {e}")

    with fb_col2:
        if st.button("👎 -1", disabled=disabled_feedback, help="Negative feedback for the last answer"):
            try:
                st.session_state.count -= 1
                save_feedback(st.session_state.last_conversation_id, -1)
                st.info("Feedback (-1) recorded.")
                print_log("Negative feedback saved to database")
            except Exception as e:
                st.error(f"Failed to save feedback: {e}")

    with fb_col3:
        st.write(f"Current count: {st.session_state.count}")
        if disabled_feedback:
            st.caption("No saved conversation yet. Ask a question first.")

    # ======== 最近的对话历史 ========
    st.subheader("Recent Conversations")
    relevance_filter = st.selectbox(
        "Filter by relevance:",
        ["All", "RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"],
        index=0,
    )
    try:
        recent_conversations = get_recent_conversations(
            limit=5,
            relevance=None if relevance_filter == "All" else relevance_filter,
        )
        if recent_conversations:
            for conv in recent_conversations:
                st.markdown(
                    f"**Q:** {conv.get('question', '')}\n\n"
                    f"**A:** {conv.get('answer', '')}\n\n"
                    f"Relevance: {conv.get('relevance', 'N/A')} | "
                    f"Model: {conv.get('model_used', 'N/A')} | "
                    f"Time: {conv.get('timestamp', '')}"
                )
                st.write("---")
        else:
            st.caption("No recent conversations.")
    except Exception as e:
        st.error(f"Failed to load recent conversations: {e}")

    # ======== 反馈统计 ========
    st.subheader("Feedback Statistics")
    try:
        feedback_stats = get_feedback_stats() or {}
        st.write(f"👍 Thumbs up: {feedback_stats.get('thumbs_up', 0) or 0}")
        st.write(f"👎 Thumbs down: {feedback_stats.get('thumbs_down', 0) or 0}")
    except Exception as e:
        st.error(f"Failed to load feedback stats: {e}")

    print_log("Streamlit app loop completed")


if __name__ == "__main__":
    print_log("Course Assistant application started")
    main()
