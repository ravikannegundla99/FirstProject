"""
main9.py — Logging, Monitoring & Conversation Persistence

Purpose:
- Demonstrate comprehensive logging with structured data (timestamps, performance metrics, errors)
- Save conversations to JSON files for debugging and analysis
- Integrate with LangSmith for advanced monitoring and chain observability
- Track token usage, response times, and conversation patterns

Key blocks:
- Structured logging with JSON format for easy parsing
- Conversation persistence to timestamped JSON files
- LangSmith integration for chain monitoring
- Performance metrics tracking (latency, token counts, cache hits)
- Error logging with context and stack traces
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

# Optional LangSmith integration (requires LANGCHAIN_API_KEY in .env)
try:
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    print("LangSmith not available. Install with: pip install langsmith")


class ConversationLogger:
    """Handles structured logging and conversation persistence"""
    
    def __init__(self, log_root: str = "logs", conversations_root: str = "conversations"):
        # Date-based folder structure
        now = datetime.now()
        self.date_str = now.strftime("%Y%m%d")
        self.hourmin_str = now.strftime("%H%M")

        self.log_dir = Path(log_root) / self.date_str
        self.conversations_dir = Path(conversations_root) / self.date_str

        # Create directories if they don't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

        # Setup structured logging (per HHMM file)
        self._setup_logging()

        # Current conversation data
        self.current_conversation: List[Dict[str, Any]] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_tag = os.getenv("SESSION_TAG", "")
        self.interaction_index: int = 0
        
    def _setup_logging(self):
        """Configure structured logging with JSON format and HHMM file names"""
        log_file = self.log_dir / f"chat_{self.date_str}_{self.hourmin_str}.log"
        
        # Create logger
        self.logger = logging.getLogger("chat_session")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler with JSON formatting and UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # JSON formatter: we will pass JSON strings as the message field
        json_line_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(json_line_formatter)

        # Console formatter with timestamp
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_interaction(self, user_input: str, bot_response: str,
                        response_time: float, token_count: Optional[int] = None,
                        cache_hit: bool = False, error: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        """Log a complete interaction including user input and bot output"""

        self.interaction_index += 1
        interaction_data: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "date": self.date_str,
            "hourmin": self.hourmin_str,
            "session_id": self.session_id,
            "session_tag": self.session_tag,
            "interaction_index": self.interaction_index,
            "user_input": user_input,
            "bot_response": bot_response,
            "response_time_seconds": round(response_time, 3),
            "token_count": token_count,
            "cache_hit": cache_hit,
            "error": error,
            "metadata": metadata or {}
        }

        # Add to current conversation
        self.current_conversation.append(interaction_data)

        # Clean error message for logging (remove emojis to avoid encoding issues)
        if error:
            # Remove common emojis from error messages
            clean_error = str(error).replace("❌", "[ERROR]").replace("⚠️", "[WARNING]").replace("✅", "[SUCCESS]")
            interaction_data["error"] = clean_error
            interaction_data["bot_response"] = clean_error
        
        # Log JSON line to file
        log_line = json.dumps({"level": "ERROR" if error else "INFO", **interaction_data}, ensure_ascii=False)
        if error:
            self.logger.error(log_line)
        else:
            self.logger.info(log_line)

        # Also write per-interaction JSON for quick diffing
        session_dir = self.conversations_dir / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        interaction_file = session_dir / f"{self.interaction_index:04d}.json"
        with open(interaction_file, 'w', encoding='utf-8') as f:
            json.dump(interaction_data, f, indent=2, ensure_ascii=False)
    
    def save_conversation(self):
        """Save current conversation to JSON file"""
        if not self.current_conversation:
            return
            
        filename = f"conversation_{self.session_id}.json"
        filepath = self.conversations_dir / filename
        
        conversation_data = {
            "session_id": self.session_id,
            "session_tag": self.session_tag,
            "start_time": self.current_conversation[0]["timestamp"] if self.current_conversation else None,
            "end_time": datetime.now().isoformat(),
            "total_interactions": len(self.current_conversation),
            "interactions": self.current_conversation
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(json.dumps({
            "level": "INFO",
            "event": "conversation_saved",
            "path": str(filepath),
            "session_id": self.session_id,
            "total_interactions": len(self.current_conversation),
            "session_tag": self.session_tag
        }))
        print(f"💾 Conversation saved to {filepath}")


class MonitoringCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for monitoring and logging"""
    
    def __init__(self, conversation_logger: ConversationLogger):
        self.conversation_logger = conversation_logger
        self.start_time = None
        self.cache_hit = False
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Called when LLM starts processing"""
        self.start_time = time.time()
        log = {
            "level": "INFO",
            "event": "llm_start",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.conversation_logger.session_id,
            "prompts_count": len(prompts)
        }
        self.conversation_logger.logger.info(json.dumps(log))
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes processing"""
        if self.start_time:
            response_time = time.time() - self.start_time
            self.cache_hit = getattr(response, "_is_cached", False)
            
            # Extract token count if available
            token_count = None
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})
                token_count = usage.get('total_tokens')
            
            log = {
                "level": "INFO",
                "event": "llm_end",
                "timestamp": datetime.now().isoformat(),
                "session_id": self.conversation_logger.session_id,
                "response_time_seconds": round(response_time, 3),
                "cache_hit": self.cache_hit,
                "token_count": token_count
            }
            self.conversation_logger.logger.info(json.dumps(log))
    
    def on_llm_error(self, error: Exception, **kwargs):
        """Called when LLM encounters an error"""
        log = {
            "level": "ERROR",
            "event": "llm_error",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.conversation_logger.session_id,
            "error": str(error)
        }
        self.conversation_logger.logger.error(json.dumps(log))


def setup_langsmith():
    """Setup LangSmith integration if available and configured"""
    if not LANGSMITH_AVAILABLE:
        return None
    
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    if not langchain_api_key:
        print("⚠️ LangSmith available but LANGCHAIN_API_KEY not set in .env")
        return None
    
    # Set LangSmith environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # Use AI Chat Pro as the project name for professional branding
    os.environ["LANGCHAIN_PROJECT"] = "AI_Chat_Pro"
    
    try:
        client = Client()
        # LangSmith integration enabled
        return client
    except Exception as e:
        print(f"❌ LangSmith setup failed: {e}")
        return None


def main():
    """Main function with comprehensive logging and monitoring"""
    
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ OpenAI API key is not available")
        return
    
    # Initialize conversation logger under provider-specific roots (OpenAI)
    conversation_logger = ConversationLogger(
        log_root=str(Path("logs") / "openai"),
        conversations_root=str(Path("conversations") / "openai")
    )
    conversation_logger.logger.info(json.dumps({
        "level": "INFO",
        "event": "session_start",
        "timestamp": datetime.now().isoformat(),
        "session_id": conversation_logger.session_id,
        "date": conversation_logger.date_str,
        "hourmin": conversation_logger.hourmin_str
    }))
    
    # Setup LangSmith if available
    langsmith_client = setup_langsmith()
    
    # Enable SQLite cache for performance monitoring
    print("📊 Using SQLite cache for performance tracking")
    set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))
    
    # Initialize model with monitoring callback (allow override via OPENAI_MODEL)
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    chat = ChatOpenAI(
        model=openai_model,
        max_tokens=1000,
        temperature=0,
        callbacks=[MonitoringCallbackHandler(conversation_logger)]
    )
    
    # System message for consistent behavior
    system_message = SystemMessage(content="You are a helpful assistant with comprehensive logging and monitoring capabilities.")
    conversation = [system_message]
    
    print(f"🤖 Chat Bot with Logging & Monitoring (type 'exit' to quit) — model={openai_model}")
    print("📁 Logs saved to: logs/openai/<YYYYMMDD>/chat_<YYYYMMDD>_<HHMM>.log")
    print("💾 Conversations saved to: conversations/openai/<YYYYMMDD>/conversation_<SESSION>.json")
    if langsmith_client:
        print("🔍 LangSmith monitoring: https://smith.langchain.com")
    
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.strip().lower() == "exit":
                print("👋 Goodbye!")
                break
            
            # Add user message to conversation
            conversation.append(HumanMessage(content=user_input))
            
            try:
                # Track response time
                start_time = time.time()
                response = chat.invoke(conversation)
                end_time = time.time()
                
                response_time = end_time - start_time
                
                # Log the interaction
                token_count = None
                if hasattr(response, 'response_metadata'):
                    usage = response.response_metadata.get('token_usage', {})
                    token_count = usage.get('total_tokens')
                conversation_logger.log_interaction(
                    user_input=user_input,
                    bot_response=response.content,
                    response_time=response_time,
                    token_count=token_count,
                    cache_hit=getattr(response, "_is_cached", False)
                )
                
                # Print response with timing info
                print(f"Bot ({response_time:.2f}s): {response.content}")
                
                # Add bot response to conversation context
                conversation.append(response)
                
            except Exception as e:
                # Log error with context
                conversation_logger.log_interaction(
                    user_input=user_input,
                    bot_response="",
                    response_time=0,
                    error=str(e)
                )
                print(f"❌ Error: {e}")
                conversation_logger.logger.error(json.dumps({
                    "level": "ERROR",
                    "event": "chat_error",
                    "timestamp": datetime.now().isoformat(),
                    "session_id": conversation_logger.session_id,
                    "error": str(e)
                }), exc_info=True)
    
    except KeyboardInterrupt:
        print("\n🛑 Session interrupted by user")
        conversation_logger.logger.info("Session interrupted by user")
    
    finally:
        # Save conversation before exiting
        conversation_logger.save_conversation()
        conversation_logger.logger.info(json.dumps({
            "level": "INFO",
            "event": "session_end",
            "timestamp": datetime.now().isoformat(),
            "session_id": conversation_logger.session_id
        }))


if __name__ == "__main__":
    main()
