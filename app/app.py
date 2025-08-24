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

# 预定义的标签和人群选项（根据你的数据）
TOPIC_OPTIONS = [
    "None",
    "medication",
    "diagnosis",
    "treatment",
    "support",
    "diet",
    "monitoring"
]

TARGET_GROUP_OPTIONS = [
    "None",
    "children_under_5",
    "children_5_and_over",
    "young_people",
    "adults"
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

    # ======== 侧边栏：模型选择和高级过滤 ========
    with st.sidebar:
        st.header("Model")
        model_choice = st.selectbox(
            "OpenAI Model",
            OPENAI_MODELS,
            index=0,
            help="Only OpenAI models are used",
        )
        
        st.header("Advanced Filters")
        
        # TOPK 设置
        st.subheader("Search Settings")
        topk_value = st.slider(
            "Number of documents to retrieve (TOPK)",
            min_value=1,
            max_value=10,
            value=5,
            help="Adjust how many documents would be used"
        )
        
        st.subheader("Topic Filter")
        topic_filter = st.multiselect(
            "Select Topics",
            TOPIC_OPTIONS,
            default=[],
            help="Filter by content topics"
        )
        topic_filter = None if "None" in topic_filter or not topic_filter else topic_filter
        
        st.subheader("Target Group Filter")
        target_group_filter = st.multiselect(
            "Select Target Groups",
            TARGET_GROUP_OPTIONS,
            default=[],
            help="Filter by target groups"
        )
        target_group_filter = None if "None" in target_group_filter or not target_group_filter else target_group_filter

    # ======== 主区：提问与回答 ========
    with st.form("qa_form", clear_on_submit=False):
        user_input = st.text_input(
            "Ask a question:",
            placeholder="e.g., What are the diagnostic criteria for adult ADHD?"
        )
        
        # 显示当前激活的过滤器
        active_filters = []
        if topic_filter:
            active_filters.append(f"Topics: {', '.join(topic_filter)}")
        if target_group_filter:
            active_filters.append(f"Target Groups: {', '.join(target_group_filter)}")
            
        if active_filters:
            st.caption(f"Active filters: {', '.join(active_filters)}")
            st.caption(f"Retrieving {topk_value} documents")
        
        submitted = st.form_submit_button("Ask")

    if submitted:
        if not user_input.strip():
            st.warning("Please enter a question before asking.")
        else:
            print_log(f"User asked: '{user_input}'")
            with st.spinner("Thinking..."):
                print_log(f"Getting answer using {model_choice} with TOPK={topk_value}")
                t0 = time.time()
                answer_data = get_answer(
                    query=user_input,
                    section_title=None,  # 移除了section过滤
                    tags=topic_filter,
                    population=target_group_filter,
                    model_choice=model_choice,
                    search_limit=topk_value,      # 使用用户设置的TOPK值
                    evaluate=False                # 关闭自动评估
                )
                t1 = time.time()
                print_log(f"Answer received in {t1 - t0:.2f} seconds")

            # 展示答案与响应时间
            st.markdown("**Answer:**")
            st.write(answer_data.get("answer", ""))

            # 显示性能指标
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Response time (s)", f"{answer_data.get('response_time', 0):.2f}")
            with col2:
                st.metric("Documents retrieved", answer_data.get("search_results_count", 0))
            with col3:
                st.metric("Cost ($)", f"{answer_data.get('openai_cost', 0):.4f}")

            # 显示搜索结果
            if answer_data.get("search_results_count", 0) > 0:
                with st.expander(f"View {answer_data['search_results_count']} source documents"):
                    for i, result in enumerate(answer_data.get("search_results", [])):
                        st.markdown(f"**Document {i+1}** (Score: {result.get('score', 0):.3f})")
                        st.markdown(f"**Source:** {result.get('source', 'N/A')}")
                        st.markdown(f"**Section:** {result.get('section_title', 'N/A')}")
                        st.markdown(f"**Topics:** {', '.join(result.get('tags', []))}")
                        st.markdown(f"**Target Groups:** {', '.join(result.get('population', []))}")
                        st.markdown("---")

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