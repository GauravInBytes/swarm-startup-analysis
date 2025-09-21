# --------------------------------------------------------------
# genai_document_assistant.py
# --------------------------------------------------------------
cat > genai_document_assistant.py << 'EOF'
import os
import json
import tempfile
from datetime import datetime
from typing import List, Dict, Any

# GCP clients
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel


# ------------------------------------------------------------------
# Helper that extracts raw media into plain‑text.
# ------------------------------------------------------------------
# Expected public methods (all return plain‑text strings):
#   - extract_from_audio(gcs_uri)
#   - extract_from_video(gcs_uri)
#   - extract_from_pdf(local_path)
#   - extract_from_docx(local_path)
#   - extract_from_pptx(local_path)
#   - extract_from_txt(local_path)   # added for plain‑text files
# ------------------------------------------------------------------
from data_extractor import DataExtractor   # <-- make sure this exists in PYTHONPATH


class GenAIDocumentAssistant:
    """
    A thin “RAG‑ish” wrapper around Gemini that works on an in‑memory
    index built from a GCS bucket.
    """

    def __init__(
        self,
        project_id: str,
        processor_id: str,
        location: str = "us-central1",
    ):
        self.project_id   = project_id
        self.processor_id = processor_id
        self.location     = location

        # ------------------------------------------------------
        # Vertex AI / Gemini initialization
        # ------------------------------------------------------
        vertexai.init(project=self.project_id, location=self.location)
        # You can change this to "gemini-2.5-pro" if you prefer
        self.model = GenerativeModel("gemini-2.5-flash")

        # ------------------------------------------------------
        # Helper services
        # ------------------------------------------------------
        self.extractor      = DataExtractor(self.project_id, self.processor_id)
        self.storage_client = storage.Client()

        # ------------------------------------------------------
        # In‑memory stores (document → text, and document → metadata)
        # ------------------------------------------------------
        self.document_contents: Dict[str, str] = {}
        self.document_metadata: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 1️⃣ Load & extract everything from a bucket
    # ------------------------------------------------------------------
    def load_documents_from_gcs(self, bucket_name: str) -> int:
        """
        Walks through every object under the bucket, extracts plain‑text
        and stores it in ``self.document_contents``.
        Returns the number of successfully loaded documents.
        """
        print("🔄 Loading documents from GCS…")
        bucket = self.storage_client.bucket(bucket_name)

        # List objects under the `extracted/` prefix
        blobs = list(bucket.list_blobs(prefix="extracted/"))
        print(f"🔎 Found {len(blobs)} objects under prefix 'extracted/'")

        for blob in blobs:
            try:
                print(f"📄 Processing: {blob.name}")

                # --------------------------------------------------
                # Determine file type & call the right extractor
                # --------------------------------------------------
                file_ext  = blob.name.split(".")[-1].lower()
                file_name = os.path.basename(blob.name)

                text_content = ""
                content_type = "Unknown"

                # ----------------- AUDIO -----------------
                if file_ext in {"wav", "mp3", "flac", "m4a"}:
                    gcs_uri = f"gs://{bucket_name}/{blob.name}"
                    text_content = self.extractor.extract_from_audio(gcs_uri)
                    content_type = "Audio Transcript"

                # ----------------- VIDEO -----------------
                elif file_ext in {"mp4", "avi", "mov", "mkv"}:
                    gcs_uri = f"gs://{bucket_name}/{blob.name}"
                    text_content = self.extractor.extract_from_video(gcs_uri)
                    content_type = "Video Transcript"

                # ----------------- PDF / DOCX / PPTX / TXT -----------------
                elif file_ext in {"pdf", "docx", "pptx", "txt"}:
                    # All of these need a local temporary file first
                    with tempfile.NamedTemporaryFile(suffix=f".{file_ext}") as tmp:
                        blob.download_to_filename(tmp.name)

                        if file_ext == "pdf":
                            text_content = self.extractor.extract_from_pdf(tmp.name)
                            content_type = "PDF Document"
                        elif file_ext == "docx":
                            text_content = self.extractor.extract_from_docx(tmp.name)
                            content_type = "Word Document"
                        elif file_ext == "pptx":
                            text_content = self.extractor.extract_from_pptx(tmp.name)
                            content_type = "PowerPoint Presentation"
                        elif file_ext == "txt":
                            text_content = self.extractor.extract_from_txt(tmp.name)
                            content_type = "Plain Text"

                # --------------------------------------------------
                # Store only if we got *something*
                # --------------------------------------------------
                if text_content.strip():
                    self.document_contents[file_name] = text_content
                    self.document_metadata[file_name] = {
                        "type": content_type,
                        "size_chars": len(text_content),
                        "gcs_path": f"gs://{bucket_name}/{blob.name}",
                        "processed_at": datetime.utcnow().isoformat() + "Z",
                    }
                    print(
                        f"✅ Loaded {file_name} "
                        f"({content_type}, {len(text_content)} chars)"
                    )
                else:
                    print(f"⚠️  No textual content extracted from {file_name}")

            except Exception as exc:
                print(f"❌ Error processing {blob.name}: {exc}")

        total = len(self.document_contents)
        print(f"\n📚 Total documents loaded: {total}")
        return total

    # ------------------------------------------------------------------
    # 2️⃣ Build a relevance‑based context for a query
    # ------------------------------------------------------------------
    def create_context_from_documents(
        self, query: str, max_context_len: int = 8000
    ) -> str:
        """
        Very simple keyword‑match relevance scorer.
        Returns a string that will be appended to the LLM prompt.
        """
        if not self.document_contents:
            return "No documents loaded."

        query_terms = set(query.lower().split())
        ranked: List[Dict[str, Any]] = []

        # Score each document
        for doc_name, content in self.document_contents.items():
            content_lower = content.lower()
            score = sum(1 for term in query_terms if term in content_lower)
            if score > 0:
                ranked.append(
                    {
                        "name": doc_name,
                        "content": content,
                        "score": score,
                        "type": self.document_metadata[doc_name]["type"],
                    }
                )

        # Fallback if nothing matches
        if not ranked:
            sample = list(self.document_contents.items())[:2]
            ranked = [
                {
                    "name": n,
                    "content": c,
                    "score": 0,
                    "type": self.document_metadata[n]["type"],
                }
                for n, c in sample
            ]

        ranked.sort(key=lambda d: d["score"], reverse=True)

        context = "DOCUMENT CONTENTS:\n\n"
        current_len = len(context)

        for doc in ranked:
            header = f"=== {doc['name']} ({doc['type']}) ===\n"
            remaining = max_context_len - current_len - len(header) - 100
            if remaining <= 0:
                break
            snippet = doc["content"][:remaining]
            context += header + snippet + "\n\n"
            current_len = len(context)

        return context

    # ------------------------------------------------------------------
    # 3️⃣ Ask a question (public method)
    # ------------------------------------------------------------------
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Returns a dict with answer, sources, confidence, etc.
        """
        if not self.document_contents:
            return {
                "answer": "No documents are loaded. Please load documents first.",
                "sources": [],
                "confidence": "low",
            }

        # 1️⃣ Build the context
        context = self.create_context_from_documents(question)

        # 2️⃣ Prompt
        prompt = f"""You are an AI assistant that answers questions strictly using the
documents provided in the CONTEXT section below. Do NOT hallucinate.
If the answer can't be found, explicitly say so and cite the relevant documents.

QUESTION: {question}

CONTEXT:
{context}

ANSWER:"""

        # 3️⃣ Call Gemini
        try:
            response = self.model.generate_content(prompt)
            answer_text = response.text.strip()
            sources = list(self.document_contents.keys())
            return {
                "answer": answer_text,
                "sources": sources,
                "confidence": "high",
                "context_length": len(context),
                "documents_used": len(sources),
            }
        except Exception as exc:
            return {
                "answer": f"Error generating response: {exc}",
                "sources": [],
                "confidence": "error",
            }

    # ------------------------------------------------------------------
    # Optional helper utilities (summary, list, search)
    # ------------------------------------------------------------------
    def get_document_summary(self) -> Dict[str, Any]:
        """Generates a concise, bullet‑point summary of all loaded docs."""
        if not self.document_contents:
            return {"summary": "No documents loaded"}

        summary_context = self.create_context_from_documents(
            "summary overview", max_context_len=12000
        )
        prompt = f"""Please provide a concise, bullet‑point summary of the following
documents. Include for each document:
• Title / file name
• Main topics covered
• Any especially important fact or figure

CONTENTS:
{summary_context}

SUMMARY:"""
        try:
            resp = self.model.generate_content(prompt)
            return {
                "summary": resp.text.strip(),
                "document_count": len(self.document_contents),
                "total_chars": sum(len(c) for c in self.document_contents.values()),
                "document_types": list(
                    set(m["type"] for m in self.document_metadata.values())
                ),
            }
        except Exception as exc:
            return {"summary": f"Error generating summary: {exc}"}

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return a lightweight list of all loaded docs + metadata."""
        return [
            {
                "name": name,
                "type": meta["type"],
                "size_chars": meta["size_chars"],
                "processed_at": meta["processed_at"],
                "gcs_path": meta["gcs_path"],
            }
            for name, meta in self.document_metadata.items()
        ]

    def search_documents(self, term: str) -> List[Dict[str, Any]]:
        """Simple substring search that returns a snippet around each hit."""
        term_lc = term.lower()
        hits: List[Dict[str, Any]] = []
        for name, content in self.document_contents.items():
            content_lc = content.lower()
            idx = content_lc.find(term_lc)
            if idx != -1:
                start = max(0, idx - 120)
                end = min(len(content), idx + len(term) + 120)
                hits.append(
                    {
                        "document": name,
                        "type": self.document_metadata[name]["type"],
                        "snippet": content[start:end],
                        "position": idx,
                    }
                )
        return hits