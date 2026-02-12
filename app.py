import streamlit as st
from src.rag_chat import load_rag

st.set_page_config(page_title="Company Policy Assistant")

st.title("Company Policy Assistant")

# -------------------
# Load model (cached)
# -------------------
@st.cache_resource
def get_qa():
    return load_rag()

qa = get_qa()

# -------------------
# Chat history
# -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -------------------
# Chat input
# -------------------
question = st.chat_input("Ask your question...")

if question:

    # show user msg
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # get answer
    result = qa.invoke({"query": question})
    st.write("DEBUG:", result["source_documents"]) 
    answer = result["result"]

    # show assistant
    with st.chat_message("assistant"):
        st.write(answer)

        if result["source_documents"]:
            st.markdown("**Sources:**")
            for doc in result["source_documents"]:
                st.write("📄", doc.metadata.get("source", "unknown"))
        else:
            st.write("No sources found.")


    st.session_state.messages.append({"role": "assistant", "content": answer})
