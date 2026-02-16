import streamlit as st
from src.rag_chat import load_rag
qa = load_rag()


st.set_page_config(page_title="Agentic Company Policy Assistant")
st.title("Agentic Company Policy Assistant")

# -------------------
# Chat history
# -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -------------------
# Chat input
# -------------------
question = st.chat_input("Ask your policy question...")

if question:

    # User message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.write(question)

    # 🔥 Agentic RAG call
    result = qa.invoke({"query": question})


    answer = result["answer"]
    policy_type = result["policy_type"]
    sources = result["sources"]

    # Assistant message
    with st.chat_message("assistant"):
        st.write(answer)
        st.markdown(f"**Policy Type:** `{policy_type}`")

        if sources:
            st.markdown("**Sources:**")
            for src in sources:
                st.write("📄", src)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
