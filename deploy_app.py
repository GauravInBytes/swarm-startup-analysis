# --------------------------------------------------------------
# deploy_app.py – Streamlit RAG app on GCS documents
# --------------------------------------------------------------
import streamlit as st
from genai_document_assistant import GenAIDocumentAssistant

# ------------------------------------------------------------------
# Fill in your project / processor IDs and bucket name
# ------------------------------------------------------------------
PROJECT_ID   = "swarm-startup-evaluator"
PROCESSOR_ID = "4b245d8abe91f49c"   # replace if needed
LOCATION     = "us-central1"
BUCKET_NAME  = "swarm-rag-bucket"

# ------------------------------------------------------------------
# Cache the assistant so it only initializes once
# ------------------------------------------------------------------
@st.cache_resource
def load_assistant():
    assistant = GenAIDocumentAssistant(
        project_id=PROJECT_ID,
        processor_id=PROCESSOR_ID,
        location=LOCATION,
    )
    num = assistant.load_documents_from_gcs(BUCKET_NAME)
    return assistant, num

# ------------------------------------------------------------------
# Main Streamlit app
# ------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Swarm-Minds RAG Assistant",
        page_icon="🧠",
        layout="wide"
    )

    st.title("🧠 Swarm-Minds RAG Assistant")
    st.write("Ask me anything about the documents in the GCS bucket.")

    # Initialize and load documents
    with st.spinner("Loading documents from GCS…"):
        assistant, num_docs = load_assistant()

    st.success(f"✅ Loaded {num_docs} documents from bucket `{BUCKET_NAME}`")

    # Sidebar with document overview
    with st.sidebar:
        st.header("📋 Document Overview")
        st.write(f"**Project:** {PROJECT_ID}")
        st.write(f"**Location:** {LOCATION}")
        st.write(f"**Bucket:** {BUCKET_NAME}")
        st.write(f"**Documents loaded:** {num_docs}")

        if st.button("📑 Show loaded documents"):
            docs = assistant.list_documents()
            st.json(docs)

        if st.button("📊 Show summary"):
            summary = assistant.get_document_summary()
            st.markdown(summary["summary"])

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I’m your RAG assistant. I’ve loaded documents from the bucket. What would you like to know?"
        })

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents and generating answer..."):
                answer = assistant.ask_question(prompt)
                st.markdown(answer["answer"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer["answer"]}
                )

                # Expandable context and sources
                with st.expander("📚 Sources used"):
                    for src in answer.get("sources", []):
                        st.write("•", src)
                with st.expander("🔎 Context length"):
                    st.write(answer.get("context_length", "N/A"))

if __name__ == "__main__":
    main()
