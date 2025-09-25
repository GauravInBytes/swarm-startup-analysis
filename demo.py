# --------------------------------------------------------------
# demo.py – quick interactive demo
# --------------------------------------------------------------
#cat > demo.py << 'EOF'
from genai_document_assistant import GenAIDocumentAssistant

# ------------------------------------------------------------------
# Fill in your project / processor IDs
# ------------------------------------------------------------------
PROJECT_ID   = "swarm-startup-evaluator"
PROCESSOR_ID = "4b245d8abe91f49c"   # <-- replace if different

# ------------------------------------------------------------------
# Instantiate the assistant
# ------------------------------------------------------------------
assistant = GenAIDocumentAssistant(
    project_id=PROJECT_ID,
    processor_id=PROCESSOR_ID,
    location="us-central1",
)

# ------------------------------------------------------------------
# Load the extracted docs from the bucket
# ------------------------------------------------------------------
NUM = assistant.load_documents_from_gcs("swarm-rag-bucket")
print(f"\nLoaded {NUM} documents.\n")

# ------------------------------------------------------------------
# Interactive Q&A loop
# ------------------------------------------------------------------
print("\n=== Ask anything about the loaded documents (type 'exit' to quit) ===")
while True:
    q = input("\nQuestion: ").strip()
    if q.lower() in {"exit", "quit"}:
        break
    answer = assistant.ask_question(q)

    print("\n--- Answer ----------------------------------------------------")
    print(answer["answer"])

    print("\n--- Sources ---------------------------------------------------")
    for src in answer["sources"][:5]:   # show first few sources only
        print(" •", src)

    print("\n-------------------------------------------------------------")
