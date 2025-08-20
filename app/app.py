import streamlit as st
import time
import uuid

from assistant import get_answer
from db import (
    save_conversation,
    save_feedback,
    get_recent_conversations,
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


def main():
    print_log("Starting your spiritual exploration journey")
    st.set_page_config(page_title="Psycho Compass", page_icon="🧠")
    st.title("🧠 Psycho Compass")

    init_states()

    # ======== 侧边栏：仅保留模型选择 ========
    with st.sidebar:
        st.header("Model")
        model_choice = st.selectbox(
            "OpenAI Model",
            OPENAI_MODELS,
            index=0,
            help="Only OpenAI models are used",
        )

    st.header("Filter")
    section_filter = st.selectbox(
        "Select Section",
        [
            "None",
            "1.3 Diagnosis",
            "1.4 Information and support",
            "1.5 Managing ADHD",
            "1.6 Dietary advice",
            "1.7 Medication",
            "1.8 Maintenance and monitoring",
        ],
        index=0,
        help="Filter guideline content by section"
    )
    section_filter = None if section_filter == "None" else section_filter

    # ======== 主区：提问与回答 ========
    with st.form("qa_form", clear_on_submit=False):
        user_input = st.text_input(
            "Ask a question:",
            placeholder="e.g., What are the diagnostic criteria for adult ADHD?"
        )
        submitted = st.form_submit_button("Ask")

    if submitted:
        if not user_input.strip():
            st.warning("Please enter a question before asking.")
        else:
            print_log(f"User asked: '{user_input}'")
            with st.spinner("Thinking..."):
                print_log(f"Getting answer using {model_choice}")
                t0 = time.time()
                answer_data = get_answer(
                    query=user_input,
                    section=section_filter,   
                    model_choice=model_choice,
                    search_limit=5,              # 固定 topK
                    evaluate=False               # 关闭自动评估
                )
                t1 = time.time()
                print_log(f"Answer received in {t1 - t0:.2f} seconds")

            # 展示答案与响应时间
            st.markdown("**Answer:**")
            st.write(answer_data.get("answer", ""))

            st.metric("Response time (s)", f"{answer_data.get('response_time', 0):.2f}")

            # 保存到数据库
            try:
                print_log("Saving conversation to database")
                save_conversation(st.session_state.conversation_id, user_input, answer_data, "adhd_guideline")
                st.session_state.last_conversation_id = st.session_state.conversation_id
                print_log("Conversation saved successfully")

                # 生成新ID供下一次提问使用
                st.session_state.conversation_id = str(uuid.uuid4())
            except Exception as e:
                st.error(f"Failed to save conversation: {e}")

    # ======== 反馈区域（仅 👍 / 👎 与感谢语） ========
    st.subheader("Feedback")
    disabled_feedback = st.session_state.last_conversation_id is None
    fb_col1, fb_col2 = st.columns([1, 1])

    feedback_given = False
    with fb_col1:
        if st.button("👍", disabled=disabled_feedback):
            try:
                save_feedback(st.session_state.last_conversation_id, 1)
                feedback_given = True
                print_log("Positive feedback saved to database")
            except Exception as e:
                st.error(f"Failed to save feedback: {e}")

    with fb_col2:
        if st.button("👎", disabled=disabled_feedback):
            try:
                save_feedback(st.session_state.last_conversation_id, -1)
                feedback_given = True
                print_log("Negative feedback saved to database")
            except Exception as e:
                st.error(f"Failed to save feedback: {e}")

    if feedback_given:
        st.success("Thanks for your feedback")

    # ======== 最近的对话历史（保留） ========
    st.subheader("Recent Conversations")
    try:
        recent_conversations = get_recent_conversations(limit=5)
        if recent_conversations:
            for conv in recent_conversations:
                st.markdown(
                    f"**Q:** {conv.get('question', '')}\n\n"
                    f"**A:** {conv.get('answer', '')}\n\n"
                    f"Time: {conv.get('timestamp', '')}"
                )
                st.write("---")
        else:
            st.caption("No recent conversations.")
    except Exception as e:
        st.error(f"Failed to load recent conversations: {e}")

    print_log("Streamlit app loop completed")


if __name__ == "__main__":
    print_log("ADHD Assistant started")
    main()
