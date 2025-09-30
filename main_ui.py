"""
Multi-Provider AI Chat UI with Streamlit

This UI integrates all 4 AI providers (OpenAI, Anthropic, Google, Groq) 
with a unified interface for easy comparison and testing.

Features:
- Provider selection dropdown
- Dynamic model selection based on provider
- Chat interface with conversation history
- Response time tracking
- Integration with existing logging system
- Real-time streaming responses
"""

import streamlit as st
import os
import time
import json
import io
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_AVAILABLE = True
    print("✅ PDF generation available - reportlab imported successfully")
except ImportError as e:
    PDF_AVAILABLE = False
    print(f"❌ PDF generation not available. Import error: {e}")
    print("Install with: pip install reportlab")

# Import our existing modules
from main9 import ConversationLogger, MonitoringCallbackHandler, setup_langsmith
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

# Load environment variables
load_dotenv()

# Model configurations
MODEL_CONFIGS = {
    "OpenAI": {
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "default": "gpt-4o-mini",
        "class": ChatOpenAI,
        "api_key": "OPENAI_API_KEY",
        "log_root": "logs/openai",
        "conv_root": "conversations/openai"
    },
    "Anthropic": {
        "models": ["claude-3-5-haiku-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229"],
        "default": "claude-3-5-haiku-20241022",
        "class": ChatAnthropic,
        "api_key": "ANTHROPIC_API_KEY",
        "log_root": "logs/anthropic",
        "conv_root": "conversations/anthropic"
    },
    "Google": {
        "models": ["gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.0-pro"],
        "default": "gemini-2.5-pro",
        "class": ChatGoogleGenerativeAI,
        "api_key": "GOOGLE_API_KEY",
        "log_root": "logs/google",
        "conv_root": "conversations/google"
    },
    "Groq": {
        "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-guard-4-12b"],
        "default": "llama-3.1-8b-instant",
        "class": ChatGroq,
        "api_key": "GROQ_API_KEY",
        "log_root": "logs/groq",
        "conv_root": "conversations/groq"
    }
}

def initialize_chat_model(provider, model, temperature=0.7):
    """Initialize the chat model based on provider and model selection"""
    config = MODEL_CONFIGS[provider]
    api_key = os.getenv(config["api_key"])
    
    if not api_key:
        st.error(f"❌ {config['api_key']} not found in environment variables")
        return None
    
    try:
        if provider == "OpenAI":
            chat = config["class"](
                model=model,
                max_tokens=1000,
                temperature=temperature,
                streaming=True
            )
        elif provider == "Anthropic":
            chat = config["class"](
                model=model,
                max_tokens=1000,
                temperature=temperature,
                streaming=True
            )
        elif provider == "Google":
            chat = config["class"](
                model=model,
                max_output_tokens=4000,
                temperature=temperature,
                streaming=True
            )
        elif provider == "Groq":
            chat = config["class"](
                model=model,
                temperature=temperature,
                streaming=True
            )
        
        return chat
    except Exception as e:
        st.error(f"❌ Error initializing {provider} model: {str(e)}")
        return None

def get_bot_response(chat, user_input, conversation_history, streaming=True, retry_enabled=True, max_retries=3):
    """Get response from the chat model with optional streaming, cache detection, and retry logic"""
    # Initialize cache_hit at function level
    cache_hit = False
    
    try:
        # Create a copy of conversation history for this request
        current_conversation = conversation_history.copy()
        
        # Add user message to conversation
        current_conversation.append(HumanMessage(content=user_input))
        
        def _get_response():
            nonlocal cache_hit  # Allow access to outer scope variable
            if streaming:
                # Streaming response
                response_content = ""
                response_container = st.empty()
                
                # Get streaming response
                start_time = time.time()
                for chunk in chat.stream(current_conversation):
                    if hasattr(chunk, 'content') and chunk.content:
                        response_content += chunk.content
                        # Update the response in real-time
                        response_container.markdown(f"**🤖 Response:**\n{response_content}")
                
                end_time = time.time()
                
                # Check for cache hit (streaming doesn't always show cache hits clearly)
                # We'll detect this by response time - very fast responses are likely cached
                if end_time - start_time < 0.5:  # Less than 500ms is likely cached
                    cache_hit = True
                
                # Create a proper response object for history
                from langchain_core.messages import AIMessage
                response = AIMessage(content=response_content)
                
                return response_content, end_time - start_time, response, cache_hit
            else:
                # Non-streaming response (original behavior)
                start_time = time.time()
                response = chat.invoke(current_conversation)
                end_time = time.time()
                
                # Check for cache hit
                if hasattr(response, '_is_cached'):
                    cache_hit = response._is_cached
                elif end_time - start_time < 0.5:  # Very fast response likely cached
                    cache_hit = True
                
                # Extract response content
                if hasattr(response, 'content') and response.content:
                    bot_response = str(response.content)
                elif hasattr(response, 'text'):
                    bot_response = str(response.text)
                else:
                    bot_response = str(response)
                
                return bot_response, end_time - start_time, response, cache_hit
        
        # Use retry logic if enabled
        if retry_enabled:
            bot_response, response_time, response, cache_hit = call_with_retries(
                _get_response, max_retries=max_retries
            )
        else:
            bot_response, response_time, response, cache_hit = _get_response()
        
        # Add both user message and bot response to conversation history
        conversation_history.append(HumanMessage(content=user_input))
        conversation_history.append(response)
        
        return bot_response, response_time, None, cache_hit
        
    except Exception as e:
        # Use plain text error message to avoid Unicode issues
        error_msg = f"Error: {str(e)}"
        return error_msg, 0, str(e), False

def compare_models(user_input, selected_models, streaming=False):
    """Compare responses from multiple models side-by-side"""
    results = []
    
    for provider, model in selected_models:
        try:
            # Initialize the model
            chat = initialize_chat_model(provider, model, temperature)
            if not chat:
                results.append({
                    "provider": provider,
                    "model": model,
                    "response": "❌ Failed to initialize model",
                    "response_time": 0,
                    "error": "Model initialization failed"
                })
                continue
            
            # Get response
            start_time = time.time()
            conversation = [HumanMessage(content=user_input)]
            
            if streaming:
                response_content = ""
                for chunk in chat.stream(conversation):
                    if hasattr(chunk, 'content') and chunk.content:
                        response_content += chunk.content
                response = response_content
            else:
                response_obj = chat.invoke(conversation)
                response = str(response_obj.content) if hasattr(response_obj, 'content') else str(response_obj)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            results.append({
                "provider": provider,
                "model": model,
                "response": response,
                "response_time": response_time,
                "error": None
            })
            
        except Exception as e:
            results.append({
                "provider": provider,
                "model": model,
                "response": f"❌ Error: {str(e)}",
                "response_time": 0,
                "error": str(e)
            })
    
    return results

def generate_session_name():
    """Generate automatic session names based on time and topic"""
    now = datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = now.strftime("%m/%d")
    return f"Chat {date_str} {time_str}"

def generate_session_name_from_question(question):
    """Generate session name based on first question"""
    # Take first few words of the question
    words = question.strip().split()[:4]  # First 4 words
    question_preview = " ".join(words)
    if len(question_preview) > 30:
        question_preview = question_preview[:30] + "..."
    return f"Chat: {question_preview}"

def create_new_session():
    """Create a new session with automatic naming"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"AI_Chat_Pro_{timestamp}"
    session_name = generate_session_name()
    
    st.session_state.sessions[session_id] = {
        "session_id": session_id,
        "name": session_name,
        "conversation_history": [],
        "created_at": datetime.now().isoformat(),
        "message_count": 0
    }
    st.session_state.current_session_id = session_id
    st.session_state.session_counter += 1
    return session_id

def get_current_session():
    """Get the current session data"""
    if st.session_state.current_session_id and st.session_state.current_session_id in st.session_state.sessions:
        return st.session_state.sessions[st.session_state.current_session_id]
    return None

def save_conversation_to_file(session_data, provider):
    """Save conversation to file like main9.py"""
    from pathlib import Path
    import json
    
    # Create conversation directory structure
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    conversations_dir = Path("conversations") / provider.lower() / date_str
    conversations_dir.mkdir(parents=True, exist_ok=True)
    
    # Create session-specific directory
    session_dir = conversations_dir / session_data["session_id"]
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full conversation
    conversation_file = session_dir / "conversation.json"
    conversation_data = {
        "session_id": session_data["session_id"],
        "name": session_data["name"],
        "created_at": session_data["created_at"],
        "message_count": session_data["message_count"],
        "provider": provider,
        "conversation_history": [
            {
                "type": "human" if isinstance(msg, HumanMessage) else "ai",
                "content": msg.content,
                "timestamp": datetime.now().isoformat()
            }
            for msg in session_data["conversation_history"]
        ]
    }
    
    with open(conversation_file, 'w', encoding='utf-8') as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    
    # Save individual interactions
    for i, msg in enumerate(session_data["conversation_history"]):
        interaction_file = session_dir / f"{i+1:04d}.json"
        interaction_data = {
            "interaction_index": i + 1,
            "timestamp": datetime.now().isoformat(),
            "type": "human" if isinstance(msg, HumanMessage) else "ai",
            "content": msg.content,
            "session_id": session_data["session_id"],
            "provider": provider
        }
        
        with open(interaction_file, 'w', encoding='utf-8') as f:
            json.dump(interaction_data, f, indent=2, ensure_ascii=False)

def load_conversations_from_files(provider):
    """Load previous conversations from files"""
    from pathlib import Path
    import json
    
    conversations = []
    conversations_dir = Path("conversations") / provider.lower()
    
    if not conversations_dir.exists():
        return conversations
    
    # Find all conversation files
    for date_dir in conversations_dir.iterdir():
        if date_dir.is_dir():
            for session_dir in date_dir.iterdir():
                if session_dir.is_dir():
                    conversation_file = session_dir / "conversation.json"
                    if conversation_file.exists():
                        try:
                            with open(conversation_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                conversations.append({
                                    "session_id": data["session_id"],
                                    "name": data["name"],
                                    "created_at": data["created_at"],
                                    "message_count": data["message_count"],
                                    "provider": data["provider"],
                                    "file_path": str(conversation_file)
                                })
                        except Exception as e:
                            print(f"Error loading conversation {conversation_file}: {e}")
    
    # Sort by creation date (newest first)
    conversations.sort(key=lambda x: x["created_at"], reverse=True)
    return conversations

def search_conversations(search_query, provider=None):
    """Search across all conversations for matching content"""
    from pathlib import Path
    import json
    
    results = []
    search_query = search_query.lower()
    
    # Search in all providers or specific provider
    providers_to_search = [provider.lower()] if provider else ["openai", "anthropic", "google", "groq"]
    
    for prov in providers_to_search:
        conversations_dir = Path("conversations") / prov
        
        if not conversations_dir.exists():
            continue
            
        # Search through all conversation files
        for date_dir in conversations_dir.iterdir():
            if date_dir.is_dir():
                for session_dir in date_dir.iterdir():
                    if session_dir.is_dir():
                        conversation_file = session_dir / "conversation.json"
                        if conversation_file.exists():
                            try:
                                with open(conversation_file, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    
                                    # Search in conversation content
                                    matches = []
                                    for interaction in data.get("interactions", []):
                                        if search_query in interaction.get("user_input", "").lower() or \
                                           search_query in interaction.get("bot_response", "").lower():
                                            matches.append({
                                                "user_input": interaction.get("user_input", ""),
                                                "bot_response": interaction.get("bot_response", ""),
                                                "timestamp": interaction.get("timestamp", "")
                                            })
                                    
                                    if matches:
                                        results.append({
                                            "session_id": data["session_id"],
                                            "name": data["name"],
                                            "created_at": data["created_at"],
                                            "provider": data["provider"],
                                            "matches": matches,
                                            "match_count": len(matches)
                                        })
                            except Exception as e:
                                print(f"Error searching conversation {conversation_file}: {e}")
    
    # Sort by match count (most matches first)
    results.sort(key=lambda x: x["match_count"], reverse=True)
    return results

def setup_caching(cache_enabled, cache_type):
    """Setup caching based on user preferences"""
    if not cache_enabled:
        # Disable caching
        set_llm_cache(None)
        return
    
    try:
        if cache_type == "SQLite (Persistent)":
            # Use SQLite cache (persistent across sessions)
            cache = SQLiteCache(database_path=".langchain_cache.db")
            set_llm_cache(cache)
        else:
            # Use In-Memory cache (temporary)
            from langchain_community.cache import InMemoryCache
            cache = InMemoryCache()
            set_llm_cache(cache)
    except Exception as e:
        st.error(f"❌ Error setting up cache: {str(e)}")

def call_with_retries(fn, *args, max_retries=3, **kwargs):
    """Retry function with exponential backoff and jitter"""
    import random
    retries = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            # Check if it's a retryable error
            error_str = str(e).lower()
            is_retryable = any(keyword in error_str for keyword in [
                'rate limit', 'timeout', 'connection', 'server error', 
                'service unavailable', 'temporary', 'retry'
            ])
            
            if not is_retryable or retries >= max_retries:
                raise e
            
            # Calculate sleep time with exponential backoff + jitter
            sleep_time = (2 ** retries) + random.uniform(0, 1)
            
            # Show retry message in UI
            st.warning(f"⚠️ Error: {str(e)[:100]}... Retrying in {sleep_time:.1f}s (attempt {retries + 1}/{max_retries})")
            time.sleep(sleep_time)
            retries += 1

def calculate_cost(provider, model, input_tokens, output_tokens):
    """Calculate cost based on provider and model pricing"""
    # Approximate pricing per 1K tokens (as of 2024)
    pricing = {
        "OpenAI": {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        },
        "Anthropic": {
            "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
            "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015}
        },
        "Google": {
            "gemini-2.5-pro": {"input": 0.00125, "output": 0.005},
            "gemini-1.5-pro": {"input": 0.00125, "output": 0.005}
        },
        "Groq": {
            "llama-3.1-8b-instant": {"input": 0.0001, "output": 0.0001},
            "llama-3.3-70b-versatile": {"input": 0.0001, "output": 0.0001}
        }
    }
    
    try:
        model_pricing = pricing.get(provider, {}).get(model, {"input": 0.001, "output": 0.002})
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        return input_cost + output_cost
    except:
        return 0.0

def update_usage_stats(provider, model, input_tokens, output_tokens, cache_hit):
    """Update usage statistics"""
    if "usage_stats" not in st.session_state:
        st.session_state.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "cache_hits": 0,
            "api_calls": 0,
            "by_provider": {}
        }
    
    stats = st.session_state.usage_stats
    stats["total_requests"] += 1
    
    if cache_hit:
        stats["cache_hits"] += 1
    else:
        stats["api_calls"] += 1
        total_tokens = input_tokens + output_tokens
        stats["total_tokens"] += total_tokens
        cost = calculate_cost(provider, model, input_tokens, output_tokens)
        stats["total_cost"] += cost
        
        # Track by provider
        if provider not in stats["by_provider"]:
            stats["by_provider"][provider] = {
                "requests": 0,
                "tokens": 0,
                "cost": 0.0
            }
        stats["by_provider"][provider]["requests"] += 1
        stats["by_provider"][provider]["tokens"] += total_tokens
        stats["by_provider"][provider]["cost"] += cost

def export_conversation_json(session_data, provider):
    """Export conversation as JSON"""
    export_data = {
        "session_info": {
            "session_id": session_data["session_id"],
            "name": session_data["name"],
            "created_at": session_data["created_at"],
            "message_count": session_data["message_count"],
            "provider": provider,
            "exported_at": datetime.now().isoformat()
        },
        "conversation": [
            {
                "type": "human" if isinstance(msg, HumanMessage) else "ai",
                "content": msg.content,
                "timestamp": datetime.now().isoformat()
            }
            for msg in session_data["conversation_history"]
        ]
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def export_conversation_pdf(session_data, provider):
    """Export conversation as PDF"""
    if not PDF_AVAILABLE:
        print("PDF_AVAILABLE is False")
        return None
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, spaceAfter=30, alignment=1)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14, spaceAfter=12)
        normal_style = styles['Normal']
        
        # Build content
        story = []
        
        # Title
        story.append(Paragraph("AI Chat Pro - Conversation Export", title_style))
        story.append(Spacer(1, 12))
        
        # Session info
        story.append(Paragraph(f"<b>Session:</b> {session_data['name']}", normal_style))
        story.append(Paragraph(f"<b>Provider:</b> {provider}", normal_style))
        story.append(Paragraph(f"<b>Created:</b> {session_data['created_at']}", normal_style))
        story.append(Paragraph(f"<b>Messages:</b> {session_data['message_count']}", normal_style))
        story.append(Spacer(1, 20))
        
        # Conversation
        story.append(Paragraph("Conversation", heading_style))
        story.append(Spacer(1, 12))
        
        for msg in session_data["conversation_history"]:
            if isinstance(msg, HumanMessage):
                story.append(Paragraph(f"<b>👤 You:</b>", normal_style))
                story.append(Paragraph(msg.content, normal_style))
            else:
                story.append(Paragraph(f"<b>🤖 {provider}:</b>", normal_style))
                story.append(Paragraph(msg.content, normal_style))
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        pdf_data = buffer.getvalue()
        print(f"PDF generated successfully! Size: {len(pdf_data)} bytes")
        return pdf_data
        
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None

def export_conversation_txt(session_data, provider):
    """Export conversation as plain text"""
    lines = []
    lines.append(f"AI Chat Pro - Conversation Export")
    lines.append(f"Session: {session_data['name']}")
    lines.append(f"Provider: {provider}")
    lines.append(f"Created: {session_data['created_at']}")
    lines.append(f"Messages: {session_data['message_count']}")
    lines.append("=" * 50)
    lines.append("")
    
    for msg in session_data["conversation_history"]:
        if isinstance(msg, HumanMessage):
            lines.append(f"👤 You: {msg.content}")
        else:
            lines.append(f"🤖 {provider}: {msg.content}")
        lines.append("")
    
    return "\n".join(lines)

def export_comparison_results(comparison_input, results):
    """Export model comparison results as text"""
    lines = []
    lines.append("AI Chat Pro - Model Comparison Results")
    lines.append("=" * 50)
    lines.append(f"Prompt: {comparison_input}")
    lines.append(f"Compared on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    for i, result in enumerate(results, 1):
        lines.append(f"Model {i}: {result['provider']} - {result['model']}")
        lines.append(f"Response Time: {result['response_time']:.2f}s")
        lines.append(f"Status: {'✅ Success' if not result['error'] else '❌ Error'}")
        lines.append("Response:")
        lines.append(result['response'])
        lines.append("-" * 40)
        lines.append("")
    
    return "\n".join(lines)

def get_theme_css(theme):
    """Generate CSS based on selected theme with improved UI/UX"""
    if theme == "Dark":
        return """
        <style>
        /* Dark Theme Variables - Improved */
        :root {
            --bg-primary: #0f0f23;
            --bg-secondary: #1a1a2e;
            --bg-tertiary: #16213e;
            --bg-card: #1e2749;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-primary: #3b82f6;
            --accent-secondary: #1d4ed8;
            --accent-success: #10b981;
            --accent-danger: #ef4444;
            --accent-warning: #f59e0b;
            --border-color: #334155;
            --border-light: #475569;
            --shadow: rgba(0, 0, 0, 0.4);
            --shadow-lg: rgba(0, 0, 0, 0.6);
        }
        
        /* Main background */
        .stApp {
            background: var(--bg-primary);
            color: var(--text-primary);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: var(--bg-secondary) !important;
            border-right: 1px solid var(--border-color);
        }
        
        /* Dropdown styling - Improved */
        .stSelectbox > div > div {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            transition: all 0.2s ease;
        }
        .stSelectbox > div > div:hover {
            background: var(--bg-tertiary);
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 1px var(--accent-primary);
        }
        .stSelectbox label {
            color: var(--text-primary) !important;
            font-weight: 600;
        }
        
        /* Chat input styling - Improved */
        .stTextArea > div > div > textarea {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            color: var(--text-primary);
            font-size: 16px;
            transition: all 0.2s ease;
        }
        .stTextArea > div > div > textarea:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
        }
        .stTextArea > div > div > textarea::placeholder {
            color: var(--text-muted);
        }
        
        /* Button styling - Improved */
        .stButton > button {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background: var(--bg-tertiary);
            border-color: var(--accent-primary);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px var(--shadow);
        }
        
        /* Send & Compare Models buttons - Green */
        .stButton > button:has-text("🚀 Send"),
        .stButton > button:has-text("⚖️ Compare Models") {
            background: linear-gradient(135deg, var(--accent-success) 0%, #059669 100%) !important;
            border: none !important;
            border-radius: 12px;
            color: white !important;
            font-weight: 700;
            font-size: 16px;
            padding: 12px 24px;
            min-height: 50px;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }
        .stButton > button:has-text("🚀 Send"):hover,
        .stButton > button:has-text("⚖️ Compare Models"):hover {
            background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
        }
        
        /* Clear/Delete buttons - Light Red */
        .stButton > button:has-text("🗑️ Clear"),
        .stButton > button:has-text("🗑️ Delete Session"),
        .stButton > button:has-text("🗑️ Clear Comparison") {
            background: linear-gradient(135deg, #f87171 0%, var(--accent-danger) 100%) !important;
            border: none !important;
            border-radius: 8px;
            color: white !important;
            font-weight: 600;
            font-size: 14px;
            padding: 8px 16px;
            min-height: 40px;
            box-shadow: 0 2px 8px rgba(248, 113, 113, 0.3);
        }
        .stButton > button:has-text("🗑️ Clear"):hover,
        .stButton > button:has-text("🗑️ Delete Session"):hover,
        .stButton > button:has-text("🗑️ Clear Comparison"):hover {
            background: linear-gradient(135deg, var(--accent-danger) 0%, #dc2626 100%) !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(248, 113, 113, 0.4);
        }
        
        /* Export button styling */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%) !important;
            border: none !important;
            border-radius: 8px;
            color: white !important;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.2s ease;
        }
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%) !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(139, 92, 246, 0.3);
        }
        
        /* Send button - Big and Green */
        .stButton > button:has-text("🚀 Send") {
            background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
            border: none;
            border-radius: 12px;
            color: white !important;
            font-weight: bold;
            font-size: 16px;
            padding: 12px 24px;
            min-height: 50px;
            transition: all 0.3s ease;
        }
        .stButton > button:has-text("🚀 Send"):hover {
            background: linear-gradient(90deg, #059669 0%, #047857 100%) !important;
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(16, 185, 129, 0.3);
        }
        
        /* Clear button - Small and Red */
        .stButton > button:has-text("🗑️ Clear") {
            background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%) !important;
            border: none;
            border-radius: 8px;
            color: white !important;
            font-weight: bold;
            font-size: 14px;
            padding: 8px 16px;
            min-height: 40px;
            transition: all 0.3s ease;
        }
        .stButton > button:has-text("🗑️ Clear"):hover {
            background: linear-gradient(90deg, #dc2626 0%, #b91c1c 100%) !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(239, 68, 68, 0.3);
        }
        
        /* Response section styling */
        .chat-section {
            background: linear-gradient(90deg, #2d3748 0%, #4a5568 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            color: var(--text-primary);
            border-left: 5px solid var(--accent-primary);
        }
        
        /* History section styling */
        .history-section {
            background: linear-gradient(90deg, #2d5a27 0%, #38a169 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            color: var(--text-primary);
            border-left: 5px solid #38a169;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        }
        
        /* Session button styling */
        .stButton > button[kind="secondary"] {
            background: linear-gradient(90deg, #38a169 0%, #48bb78 100%);
            border: none;
            border-radius: 8px;
            color: var(--text-primary);
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton > button[kind="secondary"]:hover {
            background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px var(--shadow);
        }
        
        /* Delete button styling - light red */
        .stButton > button:has-text("🗑️ Delete Session") {
            background: linear-gradient(90deg, #f56565 0%, #fc8181 100%) !important;
            border: none;
            border-radius: 8px;
            color: white !important;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton > button:has-text("🗑️ Delete Session"):hover {
            background: linear-gradient(90deg, #fc8181 0%, #f56565 100%) !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px var(--shadow);
        }
        
        /* Text elements */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
        }
        
        /* Sidebar text styling */
        .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, 
        .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
            color: var(--text-primary) !important;
        }
        
        /* Sidebar content text */
        .css-1d391kg p, .css-1d391kg div, .css-1d391kg span {
            color: var(--text-primary) !important;
        }
        
        /* Main content area text */
        .main .block-container {
            color: var(--text-primary) !important;
        }
        
        .main .block-container h1, 
        .main .block-container h2, 
        .main .block-container h3,
        .main .block-container h4, 
        .main .block-container h5, 
        .main .block-container h6 {
            color: var(--text-primary) !important;
        }
        
        /* Streamlit text elements */
        .stMarkdown, .stText, .stInfo, .stSuccess, .stWarning, .stError {
            color: var(--text-primary) !important;
        }
        
        /* Metrics */
        .metric-container {
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid var(--border-color);
        }
        
        /* Export button styling */
        .stDownloadButton > button {
            background: linear-gradient(90deg, #8b5cf6 0%, #a855f7 100%) !important;
            border: none;
            border-radius: 8px;
            color: white !important;
            font-weight: bold;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        .stDownloadButton > button:hover {
            background: linear-gradient(90deg, #a855f7 0%, #9333ea 100%) !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(139, 92, 246, 0.3);
        }
        </style>
        """
    else:  # Light theme
        return """
        <style>
        /* Light Theme Variables - Improved */
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-tertiary: #f1f5f9;
            --bg-card: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #475569;
            --text-muted: #64748b;
            --accent-primary: #3b82f6;
            --accent-secondary: #1d4ed8;
            --accent-success: #10b981;
            --accent-danger: #ef4444;
            --accent-warning: #f59e0b;
            --border-color: #e2e8f0;
            --border-light: #cbd5e1;
            --shadow: rgba(0, 0, 0, 0.1);
            --shadow-lg: rgba(0, 0, 0, 0.15);
        }
        
        /* Main background */
        .stApp {
            background: var(--bg-primary);
            color: var(--text-primary);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: var(--bg-secondary) !important;
            border-right: 1px solid var(--border-color);
        }
        
        /* Dropdown styling - Improved */
        .stSelectbox > div > div {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            transition: all 0.2s ease;
        }
        .stSelectbox > div > div:hover {
            background: var(--bg-tertiary);
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 1px var(--accent-primary);
        }
        .stSelectbox label {
            color: var(--text-primary) !important;
            font-weight: 600;
        }
        
        /* Chat input styling - Improved */
        .stTextArea > div > div > textarea {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            color: var(--text-primary);
            font-size: 16px;
            transition: all 0.2s ease;
        }
        .stTextArea > div > div > textarea:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
        }
        .stTextArea > div > div > textarea::placeholder {
            color: var(--text-muted);
        }
        
        /* Button styling - Improved */
        .stButton > button {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background: var(--bg-tertiary);
            border-color: var(--accent-primary);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px var(--shadow);
        }
        
        /* Send & Compare Models buttons - Green */
        .stButton > button:has-text("🚀 Send"),
        .stButton > button:has-text("⚖️ Compare Models") {
            background: linear-gradient(135deg, var(--accent-success) 0%, #059669 100%) !important;
            border: none !important;
            border-radius: 12px;
            color: white !important;
            font-weight: 700;
            font-size: 16px;
            padding: 12px 24px;
            min-height: 50px;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }
        .stButton > button:has-text("🚀 Send"):hover,
        .stButton > button:has-text("⚖️ Compare Models"):hover {
            background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
        }
        
        /* Clear/Delete buttons - Light Red */
        .stButton > button:has-text("🗑️ Clear"),
        .stButton > button:has-text("🗑️ Delete Session"),
        .stButton > button:has-text("🗑️ Clear Comparison") {
            background: linear-gradient(135deg, #f87171 0%, var(--accent-danger) 100%) !important;
            border: none !important;
            border-radius: 8px;
            color: white !important;
            font-weight: 600;
            font-size: 14px;
            padding: 8px 16px;
            min-height: 40px;
            box-shadow: 0 2px 8px rgba(248, 113, 113, 0.3);
        }
        .stButton > button:has-text("🗑️ Clear"):hover,
        .stButton > button:has-text("🗑️ Delete Session"):hover,
        .stButton > button:has-text("🗑️ Clear Comparison"):hover {
            background: linear-gradient(135deg, var(--accent-danger) 0%, #dc2626 100%) !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(248, 113, 113, 0.4);
        }
        
        /* Export button styling */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%) !important;
            border: none !important;
            border-radius: 8px;
            color: white !important;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.2s ease;
        }
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #a855f7 0%, #9333ea 100%) !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(139, 92, 246, 0.3);
        }
        
        /* Text elements */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
        }
        
        /* Metrics */
        .metric-container {
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid var(--border-color);
        }
        
        </style>
        """

def main():
    st.set_page_config(
        page_title="AI Chat Pro",
        page_icon="🚀",
        layout="wide"
    )
    
    # Initialize session state FIRST (before any other code)
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "session_counter" not in st.session_state:
        st.session_state.session_counter = 1
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = None
    if "logger" not in st.session_state:
        st.session_state.logger = None
    if "current_config" not in st.session_state:
        st.session_state.current_config = None
    if "theme" not in st.session_state:
        st.session_state.theme = "Light"
    if "streaming" not in st.session_state:
        st.session_state.streaming = True
    if "cache_enabled" not in st.session_state:
        st.session_state.cache_enabled = True
    if "cache_type" not in st.session_state:
        st.session_state.cache_type = "SQLite (Persistent)"
    if "retry_enabled" not in st.session_state:
        st.session_state.retry_enabled = True
    if "max_retries" not in st.session_state:
        st.session_state.max_retries = 3
    
    # Create default session if none exists
    if not st.session_state.sessions:
        # Create session manually to avoid function call issues
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"AI_Chat_Pro_{timestamp}"
        session_name = generate_session_name()
        
        st.session_state.sessions[session_id] = {
            "session_id": session_id,
            "name": session_name,
            "conversation_history": [],
            "created_at": datetime.now().isoformat(),
            "message_count": 0
        }
        st.session_state.current_session_id = session_id
        st.session_state.session_counter += 1
    
    st.title("🚀 AI Chat Pro")
    st.markdown("**Compare & test responses from OpenAI, Anthropic, Google, and Groq models**")
    
    # Apply theme-based CSS
    st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Provider selection (default to Groq for faster responses)
        provider = st.selectbox(
            "Select AI Provider",
            list(MODEL_CONFIGS.keys()),
            index=list(MODEL_CONFIGS.keys()).index("Groq"),  # Set Groq as default
            help="Choose which AI provider to use"
        )
        
        # Model selection based on provider
        config = MODEL_CONFIGS[provider]
        model = st.selectbox(
            "Select Model",
            config["models"],
            index=config["models"].index(config["default"]),
            help=f"Available models for {provider}"
        )
        
        # Temperature Control
        st.subheader("🌡️ Response Settings")
        temperature = st.slider(
            "Temperature (Creativity Level)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower values = more focused/consistent responses\nHigher values = more creative/diverse responses"
        )
        
        # Temperature explanation
        if temperature <= 0.3:
            st.info("🎯 **Focused Mode**: Responses will be more deterministic and consistent")
        elif temperature <= 0.7:
            st.info("⚖️ **Balanced Mode**: Good balance of creativity and consistency")
        else:
            st.info("🎨 **Creative Mode**: Responses will be more diverse and creative")
        
        # Session settings (removed for cleaner UI - session info available in exports)
        
        # Check API key availability (hidden from UI)
        api_key = os.getenv(config["api_key"])
        if not api_key:
            st.error(f"❌ {config['api_key']} missing")
        
        # Theme Selection
        st.header("🎨 Theme")
        theme = st.selectbox(
            "Select Theme:",
            ["Light", "Dark"],
            index=0,
            help="Choose your preferred theme"
        )
        
        # Store theme preference
        if "theme" not in st.session_state:
            st.session_state.theme = "Light"
        if st.session_state.theme != theme:
            st.session_state.theme = theme
            st.rerun()
        
        # Streaming Toggle
        st.header("⚡ Response Mode")
        streaming = st.toggle(
            "Enable Streaming",
            value=True,
            help="Show responses in real-time as they're generated (like ChatGPT)"
        )
        
        # Store streaming preference
        if "streaming" not in st.session_state:
            st.session_state.streaming = True
        if st.session_state.streaming != streaming:
            st.session_state.streaming = streaming
        
        # Advanced Features (Hidden from UI - Backend Only)
        # These features are enabled by default but not shown to users
        
        # Caching - Always enabled (SQLite persistent)
        if "cache_enabled" not in st.session_state:
            st.session_state.cache_enabled = True
        if "cache_type" not in st.session_state:
            st.session_state.cache_type = "SQLite (Persistent)"
        
        # Retry Logic - Always enabled with default settings
        if "retry_enabled" not in st.session_state:
            st.session_state.retry_enabled = True
        if "max_retries" not in st.session_state:
            st.session_state.max_retries = 3
        
        # Usage Analytics - Backend tracking only
        if "usage_stats" not in st.session_state:
            st.session_state.usage_stats = {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "cache_hits": 0,
                "api_calls": 0,
                "by_provider": {}
            }
        
        # Search Functionality
        st.header("🔍 Search")
        search_query = st.text_input(
            "Search conversations:",
            placeholder="Type keywords to search...",
            help="Search across all your conversations (max 2000 characters)",
            max_chars=2000  # 2000 character limit for search
        )
        
        # Character counter for search
        if search_query:
            char_count = len(search_query)
            char_remaining = 2000 - char_count
            if char_remaining < 100:
                st.warning(f"⚠️ {char_remaining} characters remaining (Limit: 2000)")
            else:
                st.caption(f"🔍 {char_count}/2000 characters")
        
        if search_query.strip():
            with st.spinner("🔍 Searching..."):
                search_results = search_conversations(search_query, provider)
                
                if search_results:
                    st.success(f"Found {len(search_results)} conversations with matches")
                    
                    for result in search_results[:5]:  # Show top 5 results
                        with st.expander(f"📝 {result['name']} ({result['provider']}) - {result['match_count']} matches"):
                            st.write(f"**Session:** {result['name']}")
                            st.write(f"**Provider:** {result['provider']}")
                            st.write(f"**Created:** {result['created_at'][:10]}")
                            
                            for match in result['matches'][:3]:  # Show first 3 matches
                                st.write("**👤 You:**", match['user_input'][:100] + "..." if len(match['user_input']) > 100 else match['user_input'])
                                st.write("**🤖 AI:**", match['bot_response'][:100] + "..." if len(match['bot_response']) > 100 else match['bot_response'])
                                st.write("---")
                else:
                    st.info("No conversations found matching your search")
        
        # Session Management
        st.header("📋 Sessions")
        
        # Create new session button
        if st.button("➕ New Session", use_container_width=True):
            create_new_session()
            st.rerun()
        
        # Session selection
        if st.session_state.sessions:
            session_names = {sid: data["name"] for sid, data in st.session_state.sessions.items()}
            current_name = session_names.get(st.session_state.current_session_id, "Select a session")
            
            selected_name = st.selectbox(
                "Current Session:",
                options=list(session_names.values()),
                index=list(session_names.values()).index(current_name) if current_name in session_names.values() else 0,
                help="Switch between different chat sessions"
            )
            
            # Update current session
            for sid, name in session_names.items():
                if name == selected_name:
                    st.session_state.current_session_id = sid
                    break
        
        # Session info removed for cleaner UI - available in exports and JSON logs
        
        # Export section
        st.header("📤 Export")
        current_session = get_current_session()
        if current_session and current_session["conversation_history"]:
            # Get current provider for export
            current_provider = provider
            
            col1, col2 = st.columns(2)
            with col1:
                    # PDF Export (main export)
                    if PDF_AVAILABLE:
                        try:
                            pdf_data = export_conversation_pdf(current_session, current_provider)
                            if pdf_data:
                                # Generate filename with timestamp and provider
                                timestamp = current_session['created_at'].replace(':', '').replace('-', '').replace(' ', '_')
                                # For model comparison, use "Comparison" instead of provider
                                # Check if we're in comparison mode (selected_models exists and has multiple models)
                                try:
                                    is_comparison = 'selected_models' in locals() and len(selected_models) > 1
                                except NameError:
                                    is_comparison = False
                                provider_name = "Comparison" if is_comparison else current_provider
                                filename = f"AI_Chat_Pro_{timestamp}_{provider_name}_PDF.pdf"
                                
                                st.download_button(
                                    label="📄 PDF",
                                    data=pdf_data,
                                    file_name=filename,
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            else:
                                st.error("PDF generation failed - no data returned")
                        except Exception as e:
                            st.error(f"PDF generation error: {str(e)}")
                    else:
                        st.info("Install reportlab for PDF export")
                        st.caption(f"PDF_AVAILABLE: {PDF_AVAILABLE}")
            
            with col2:
                # JSON Export
                json_data = export_conversation_json(current_session, current_provider)
                # Generate filename with timestamp and provider
                timestamp = current_session['created_at'].replace(':', '').replace('-', '').replace(' ', '_')
                # For model comparison, use "Comparison" instead of provider
                # Check if we're in comparison mode (selected_models exists and has multiple models)
                try:
                    is_comparison = 'selected_models' in locals() and len(selected_models) > 1
                except NameError:
                    is_comparison = False
                provider_name = "Comparison" if is_comparison else current_provider
                json_filename = f"AI_Chat_Pro_{timestamp}_{provider_name}_JSON.json"
                
                st.download_button(
                    label="📊 JSON",
                    data=json_data,
                    file_name=json_filename,
                    mime="application/json",
                    use_container_width=True
                )
            
            # TXT Export (compact)
            txt_data = export_conversation_txt(current_session, current_provider)
            # Generate filename with timestamp and provider
            timestamp = current_session['created_at'].replace(':', '').replace('-', '').replace(' ', '_')
            # For model comparison, use "Comparison" instead of provider
            # Check if we're in comparison mode (selected_models exists and has multiple models)
            try:
                is_comparison = 'selected_models' in locals() and len(selected_models) > 1
            except NameError:
                is_comparison = False
            provider_name = "Comparison" if is_comparison else current_provider
            txt_filename = f"AI_Chat_Pro_{timestamp}_{provider_name}_Text.txt"
            
            st.download_button(
                label="📝 Text",
                data=txt_data,
                file_name=txt_filename,
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("💬 No messages to export yet")
        
        # Delete current session button
        if st.button("🗑️ Delete Session", use_container_width=True):
            if st.session_state.current_session_id in st.session_state.sessions:
                del st.session_state.sessions[st.session_state.current_session_id]
                if st.session_state.sessions:
                    st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]
                else:
                    st.session_state.current_session_id = None
                st.rerun()
        
    
    
    # Initialize chat model if provider/model/temperature changed
    current_config = f"{provider}_{model}_{temperature}"
    if f"current_config" not in st.session_state or st.session_state.current_config != current_config:
        with st.spinner(f"Initializing {provider} {model}..."):
            st.session_state.chat_model = initialize_chat_model(provider, model, temperature)
            if st.session_state.chat_model:
                # Initialize logger for this provider
                config = MODEL_CONFIGS[provider]
                st.session_state.logger = ConversationLogger(
                    log_root=config["log_root"],
                    conversations_root=config["conv_root"]
                )
                # Setup LangSmith with SESSION_TAG and UI identifier
                setup_langsmith()
                if os.getenv("LANGCHAIN_API_KEY"):
                    base_proj = os.getenv("LANGCHAIN_PROJECT", "chat-monitoring-demo")
                    session_tag = os.getenv("SESSION_TAG", "ui_session")
                    os.environ["LANGCHAIN_PROJECT"] = f"{base_proj}:{session_tag}:{provider.lower()}:ui"
                
                # Setup caching based on user preferences
                setup_caching(st.session_state.cache_enabled, st.session_state.cache_type)
                st.session_state.current_config = current_config
                st.success(f"✅ {provider} {model} initialized successfully!")
            else:
                st.error(f"❌ Failed to initialize {provider} {model}")
    
    # Main chat interface
    if st.session_state.chat_model and st.session_state.logger:
        current_session = get_current_session()
        if not current_session:
            st.error("❌ No active session")
            return
        
        # Add tabs for different modes
        tab1, tab2 = st.tabs(["💬 Single Chat", "⚖️ Model Comparison"])
        
        with tab1:
            # Chat input with character limit
            user_input = st.text_area(
                "Your message:",
                height=120,
                placeholder="Type your message here...",
                help="Enter your question or prompt for the AI model (max 4000 characters)",
                max_chars=4000  # 4000 character limit
            )
            
            # Character counter with clear limit display
            if user_input:
                char_count = len(user_input)
                char_remaining = 4000 - char_count
                if char_remaining < 200:
                    st.warning(f"⚠️ {char_remaining} characters remaining (Limit: 4000)")
                elif char_remaining < 500:
                    st.info(f"ℹ️ {char_remaining} characters remaining (Limit: 4000)")
                else:
                    st.caption(f"📝 {char_count}/4000 characters")
            else:
                st.caption("📝 0/4000 characters")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                send_button = st.button("🚀 Send", type="primary", use_container_width=True)
            with col2:
                if st.button("🗑️ Clear", use_container_width=True):
                    current_session["conversation_history"] = []
                    current_session["message_count"] = 0
                    st.rerun()
        
        with tab2:
            st.header("⚖️ Model Comparison")
            st.markdown("Compare responses from multiple AI models side-by-side")
            
            # Model selection for comparison
            st.subheader("Select Models to Compare (Select Max 3 Models)")
            
            # Get available models from all providers
            available_models = []
            for prov, config in MODEL_CONFIGS.items():
                api_key = os.getenv(config["api_key"])
                if api_key:  # Only include providers with API keys
                    for model_name in config["models"]:
                        available_models.append((prov, model_name))
            
            if available_models:
                # Multi-select for models (max 3)
                selected_models = st.multiselect(
                    "Choose models to compare:",
                    options=available_models,
                    format_func=lambda x: f"{x[0]} - {x[1]}",
                    default=available_models[:2] if len(available_models) >= 2 else available_models,
                    help="Select 2-3 models for comparison (max 3 for better UI)",
                    max_selections=3
                )
                
                # Comparison input with character limit
                comparison_input = st.text_area(
                    "Enter your prompt for comparison:",
                    height=120,
                    placeholder="Type your question or prompt here...",
                    help="This will be sent to all selected models (max 4000 characters)",
                    max_chars=4000  # 4000 character limit
                )
                
                # Character counter for comparison input with clear limit display
                if comparison_input:
                    char_count = len(comparison_input)
                    char_remaining = 4000 - char_count
                    if char_remaining < 200:
                        st.warning(f"⚠️ {char_remaining} characters remaining (Limit: 4000)")
                    elif char_remaining < 500:
                        st.info(f"ℹ️ {char_remaining} characters remaining (Limit: 4000)")
                    else:
                        st.caption(f"📝 {char_count}/4000 characters")
                else:
                    st.caption("📝 0/4000 characters")
                
                # Use global temperature setting (no duplicate control needed)
                st.info(f"🌡️ **Using Temperature**: {temperature} - {'Focused' if temperature <= 0.3 else 'Balanced' if temperature <= 0.7 else 'Creative'} Mode")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    compare_button = st.button("⚖️ Compare Models", type="primary", use_container_width=True)
                with col2:
                    if st.button("🗑️ Clear Comparison", use_container_width=True):
                        st.rerun()
                
                # Process comparison
                if compare_button and comparison_input.strip() and len(selected_models) >= 2:
                    with st.spinner("🔄 Comparing models..."):
                        results = compare_models(comparison_input, selected_models, streaming=False)
                        
                        # Display results side-by-side
                        st.subheader("📊 Comparison Results")
                        
                        # Create columns for each model
                        cols = st.columns(len(results))
                        
                        for i, result in enumerate(results):
                            with cols[i]:
                                st.markdown(f"**🤖 {result['provider']} - {result['model']}**")
                                
                                # Response time metric
                                st.metric("Response Time", f"{result['response_time']:.2f}s")
                                
                                # Response content
                                st.markdown("**Response:**")
                                st.markdown(result['response'])
                                
                                if result['error']:
                                    st.error(f"Error: {result['error']}")
                                
                                st.markdown("---")
                        
                        # Summary comparison
                        st.subheader("📈 Performance Summary")
                        summary_data = []
                        for result in results:
                            summary_data.append({
                                "Provider": result['provider'],
                                "Model": result['model'],
                                "Response Time (s)": f"{result['response_time']:.2f}",
                                "Temperature": f"{temperature}",
                                "Status": "✅ Success" if not result['error'] else "❌ Error"
                            })
                        
                        st.table(summary_data)
                        
                        # Export comparison results
                        st.subheader("📤 Export Comparison Results")
                        comparison_export_data = export_comparison_results(comparison_input, results)
                        st.download_button(
                            label="📄 Export Comparison Results",
                            data=comparison_export_data,
                            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
            else:
                st.warning("⚠️ No models available for comparison. Please configure API keys.")
        
        # Process message
        if send_button and user_input.strip():
            with st.spinner("🤔 Thinking..."):
                # Rename session based on first question if it's still the default name
                if current_session["message_count"] == 0 and current_session["name"].startswith("Chat "):
                    new_name = generate_session_name_from_question(user_input)
                    current_session["name"] = new_name
                
                # Get response
                bot_response, response_time, error, cache_hit = get_bot_response(
                    st.session_state.chat_model,
                    user_input,
                    current_session["conversation_history"],
                    streaming=st.session_state.streaming,
                    retry_enabled=st.session_state.retry_enabled,
                    max_retries=st.session_state.max_retries
                )
                
                # Log the interaction
                if st.session_state.logger:
                    st.session_state.logger.log_interaction(
                        user_input=user_input,
                        bot_response=bot_response,
                        response_time=response_time,
                        error=error
                    )
                
                # Update session message count
                current_session["message_count"] += 2  # user + bot message
                
                # Estimate tokens and update usage stats
                input_tokens = len(user_input.split()) * 1.3  # Rough estimation
                output_tokens = len(bot_response.split()) * 1.3  # Rough estimation
                update_usage_stats(provider, model, int(input_tokens), int(output_tokens), cache_hit)
                
                # Save conversation to file
                save_conversation_to_file(current_session, provider)
                
                # Display response with styling
                st.markdown('<div class="chat-section">', unsafe_allow_html=True)
                st.markdown(f"**🤖 {provider} Response:**")
                st.markdown(bot_response)
                
                # Display metrics
                st.markdown("**📊 Response Metrics:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Response Time", f"{response_time:.2f}s")
                with col2:
                    st.metric("Provider", provider)
                with col3:
                    st.metric("Model", model)
                with col4:
                    st.metric("Temperature", f"{temperature}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Keep the user input visible (don't clear it automatically)
                # This allows users to see what they asked and potentially modify it
        
        # Display conversation history with styling (only when 4+ messages - previous conversations)
        if len(current_session["conversation_history"]) >= 4:
            st.markdown('<div class="history-section">', unsafe_allow_html=True)
            st.header("📜 Previous Messages")
            
            # Show only previous messages (exclude the last 2 messages which are already shown above)
            previous_messages = current_session["conversation_history"][:-2]
            
            for i, message in enumerate(previous_messages):
                if hasattr(message, 'content'):
                    if isinstance(message, HumanMessage):
                        st.markdown(f"**👤 You:** {message.content}")
                    else:
                        st.markdown(f"**🤖 {provider}:** {message.content}")
                st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.warning("⚠️ Please configure the required API keys in your .env file")
        st.markdown("""
        **Required environment variables:**
        - `OPENAI_API_KEY` for OpenAI
        - `ANTHROPIC_API_KEY` for Anthropic  
        - `GOOGLE_API_KEY` for Google
        - `GROQ_API_KEY` for Groq
        - `SESSION_TAG` (optional)
        - `LANGCHAIN_API_KEY` (optional, for monitoring)
        """)

if __name__ == "__main__":
    main()
