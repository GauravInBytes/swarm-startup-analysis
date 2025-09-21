# Create enhanced app with RAG capabilities
cat > app.py << 'EOF'
import streamlit as st
import os
from pathlib import Path
import json

# Configure the page
st.set_page_config(
    page_title="Swarm-Minds RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

# Initialize Vertex AI with proper error handling
@st.cache_resource
def init_gemini():
    PROJECT_ID = "swarm-minds"
    REGION = "us-central1"
    
    try:
        st.info(f"Initializing with Project: {PROJECT_ID}, Region: {REGION}")
        
        # Try different import methods
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            st.success("✅ Using vertexai.generative_models")
        except ImportError:
            try:
                import vertexai
                from vertexai.preview.generative_models import GenerativeModel
                st.success("✅ Using vertexai.preview.generative_models")
            except ImportError:
                st.error("❌ No compatible Vertex AI library found")
                return None, None
        
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=REGION)
        
        # Try different model names
        models_to_try = [
            "gemini-2.5-flash",
        ]
        
        for model_name in models_to_try:
            try:
                st.info(f"Trying model: {model_name}")
                model = GenerativeModel(model_name)
                
                # Test the model
                test_response = model.generate_content("Hello")
                st.success(f"✅ Successfully initialized: {model_name}")
                return model, model_name
                
            except Exception as model_error:
                st.warning(f"❌ Failed with {model_name}: {str(model_error)}")
                continue
        
        st.error("❌ No Gemini models are available")
        return None, None
        
    except Exception as e:
        st.error(f"❌ Failed to initialize Vertex AI: {str(e)}")
        return None, None

@st.cache_data
def load_project_knowledge():
    """Load project-specific knowledge base"""
    
    # Project knowledge base - Add your project information here
    knowledge_base = {
        "project_info": {
            "name": "Swarm-Minds",
            "description": "An AI-powered project focused on collective intelligence and swarm behavior analysis",
            "technologies": ["Google Cloud", "Vertex AI", "Gemini", "Python", "Streamlit"],
            "purpose": "To create intelligent systems that can process and analyze complex data patterns"
        },
        "features": [
            "RAG (Retrieval-Augmented Generation) capabilities",
            "Real-time chat interface with Gemini AI",
            "Cloud-based deployment on Google Cloud Run",
            "Scalable architecture for handling multiple users",
            "Integration with Vertex AI for advanced AI capabilities"
        ],
        "technical_details": {
            "cloud_platform": "Google Cloud Platform",
            "ai_model": "Gemini 2.5 Flash",
            "deployment": "Cloud Run",
            "frontend": "Streamlit",
            "region": "us-central1"
        },
        "use_cases": [
            "Intelligent document analysis",
            "Question-answering systems",
            "Knowledge management",
            "AI-powered research assistance",
            "Automated content generation"
        ]
    }
    
    return knowledge_base

def search_knowledge_base(query, knowledge_base):
    """Simple knowledge base search"""
    query_lower = query.lower()
    relevant_info = []
    
    # Search in different sections
    sections_to_search = [
        ("Project Information", knowledge_base["project_info"]),
        ("Features", {"features": knowledge_base["features"]}),
        ("Technical Details", knowledge_base["technical_details"]),
        ("Use Cases", {"use_cases": knowledge_base["use_cases"]})
    ]
    
    for section_name, section_data in sections_to_search:
        section_text = str(section_data).lower()
        
        # Check for keyword matches
        keywords = ["swarm", "minds", "ai", "gemini", "cloud", "rag", "project", "features", "technical"]
        if any(keyword in query_lower for keyword in keywords):
            if any(keyword in section_text for keyword in query_lower.split()):
                relevant_info.append(f"**{section_name}:**\n{json.dumps(section_data, indent=2)}")
    
    return relevant_info

def generate_rag_response(model, query, knowledge_base):
    """Generate response using RAG approach"""
    
    # Search for relevant information
    relevant_info = search_knowledge_base(query, knowledge_base)
    
    # Create context from relevant information
    context = "\n\n".join(relevant_info) if relevant_info else "No specific project information found."
    
    # Create enhanced prompt with context
    rag_prompt = f"""You are an AI assistant for the Swarm-Minds project. Use the following project information to answer the user's question accurately and helpfully.

PROJECT CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer based primarily on the project context provided
2. If the context doesn't contain relevant information, provide a general helpful response
3. Be specific about Swarm-Minds project details when available
4. Mention that you're answering based on the Swarm-Minds project knowledge
5. Be conversational and helpful

ANSWER:"""

    return rag_prompt

def main():
    st.title("🧠 Swarm-Minds RAG Assistant")
    st.write("Ask me anything about the Swarm-Minds project! I have access to project-specific knowledge.")
    
    # Load knowledge base
    knowledge_base = load_project_knowledge()
    
    # Sidebar with project info
    with st.sidebar:
        st.header("📋 Project Overview")
        st.write(f"**Project:** {knowledge_base['project_info']['name']}")
        st.write(f"**Description:** {knowledge_base['project_info']['description']}")
        
        st.header("🛠️ Technologies")
        for tech in knowledge_base['project_info']['technologies']:
            st.write(f"• {tech}")
        
        st.header("💡 Try asking:")
        st.write("• What is Swarm-Minds?")
        st.write("• What technologies are used?")
        st.write("• What are the main features?")
        st.write("• How is it deployed?")
        st.write("• What are the use cases?")
        
        # Knowledge base viewer
        with st.expander("📚 View Knowledge Base"):
            st.json(knowledge_base)
    
    # Debug information
    with st.expander("🔧 Debug Info"):
        st.write("**Knowledge Base Status:**")
        st.write(f"✅ Loaded {len(knowledge_base)} sections")
        
        try:
            import vertexai
            st.write(f"✅ vertexai version: {vertexai.__version__}")
        except:
            st.write("❌ vertexai not found")
    
    # Initialize model
    model, model_name = init_gemini()
    
    if model is None:
        st.error("🚫 Could not initialize Gemini.")
        st.stop()
    
    # Show which model is being used
    st.success(f"🎯 Using model: **{model_name}** with RAG capabilities")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"Hello! I'm your Swarm-Minds project assistant powered by {model_name}. I have access to project-specific knowledge and can answer questions about the Swarm-Minds project. What would you like to know?"
        })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about the Swarm-Minds project..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response using RAG
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching project knowledge and generating response..."):
                try:
                    # Generate RAG prompt
                    rag_prompt = generate_rag_response(model, prompt, knowledge_base)
                    
                    # Get response from Gemini
                    response = model.generate_content(rag_prompt)
                    response_text = response.text
                    
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                    # Show what context was used (optional)
                    with st.expander("🔍 Context Used"):
                        relevant_info = search_knowledge_base(prompt, knowledge_base)
                        if relevant_info:
                            for info in relevant_info:
                                st.markdown(info)
                        else:
                            st.write("No specific project context found for this query.")
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
EOF

# Update requirements to include additional packages for RAG
cat > requirements.txt << 'EOF'
streamlit==1.49.1
google-cloud-aiplatform==1.115.0
vertexai==1.43.0
google-auth==2.40.3
google-auth-oauthlib==1.1.0
google-auth-httplib2==0.2.0
pathlib
json5
EOF

# Deploy the enhanced RAG app
echo "🚀 Deploying Swarm-Minds RAG Assistant..."
gcloud run deploy gemini-chat \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --port 8080 \
    --set-env-vars GOOGLE_CLOUD_PROJECT=swarm-minds \
    --max-instances 10 \
    --timeout 600

echo ""
echo "🎉 RAG-enabled deployment complete!"
echo "Your enhanced app URL:"
gcloud run services describe gemini-chat --region=us-central1 --format='value(status.url)'
echo ""
echo "✨ New features added:"
echo "• Project-specific knowledge base"
echo "• RAG (Retrieval-Augmented Generation)"
echo "• Context-aware responses"
echo "• Project information sidebar"
echo "• Knowledge base search"