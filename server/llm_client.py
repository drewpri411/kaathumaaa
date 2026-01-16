"""OpenAI LLM client for chat completions."""
import asyncio
from typing import List, Optional, AsyncGenerator

from openai import AsyncOpenAI

from .config import config
from .event_bus import event_bus, EventType


class LLMClient:
    """
    OpenAI API client for LLM responses.
    
    Features:
    - GPT-4o-mini model
    - Streaming response handling
    - Conversation history management
    - Timeout handling
    """
    
    def __init__(self):
        """Initialize LLM client."""
        if config.openai_api_key:
            self.client = AsyncOpenAI(api_key=config.openai_api_key)
        else:
            print("WARNING: OpenAI API key not set, LLM will not work")
            self.client = None
        
        self.model = config.llm_model
        self.max_tokens = config.llm_max_tokens
        self.temperature = config.llm_temperature
        
        # System prompt
        self.system_prompt = (
            "You are a helpful voice assistant. Keep responses concise and natural, "
            "suitable for voice conversation. Limit to 2-3 sentences. "
            "Be conversational and friendly."
        )
    
    async def generate_response(
        self,
        conversation_history: str,
        user_utterance: str
    ) -> AsyncGenerator[str, None]:
        """
        Generate response using OpenAI.
        
        Args:
            conversation_history: Recent conversation history
            user_utterance: Latest user utterance
        
        Yields:
            Response chunks
        """
        if not self.client:
            yield "I'm sorry, I'm not configured properly to respond."
            return
        
        # Build messages
        messages = self.build_messages(conversation_history, user_utterance)
        
        # Emit generating event
        await event_bus.emit(EventType.RESPONSE_GENERATING, {
            "prompt_length": len(str(messages))
        })
        
        try:
            # Generate with streaming
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )
            
            # Stream chunks
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    yield text
                    
                    # Emit chunk event
                    await event_bus.emit(EventType.RESPONSE_CHUNK, {
                        "chunk": text
                    })
        
        except Exception as e:
            print(f"LLM generation error: {e}")
            yield "I'm sorry, I encountered an error generating a response."
    
    def build_messages(self, conversation_history: str, user_utterance: str) -> List[dict]:
        """
        Build messages array from conversation history.
        
        Args:
            conversation_history: Recent conversation
            user_utterance: Latest user input
        
        Returns:
            Messages array for OpenAI API
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Parse conversation history
        if conversation_history:
            history_lines = conversation_history.strip().split('\n')
            # Get last 10 lines (5 exchanges)
            recent_history = history_lines[-10:]
            
            for line in recent_history:
                if line.startswith("User:"):
                    messages.append({
                        "role": "user",
                        "content": line.replace("User:", "").strip()
                    })
                elif line.startswith("Agent:"):
                    messages.append({
                        "role": "assistant",
                        "content": line.replace("Agent:", "").strip()
                    })
        
        # Add current user utterance
        messages.append({
            "role": "user",
            "content": user_utterance
        })
        
        return messages
    
    async def generate_response_complete(
        self,
        conversation_history: str,
        user_utterance: str
    ) -> str:
        """
        Generate complete response (non-streaming).
        
        Args:
            conversation_history: Recent conversation history
            user_utterance: Latest user utterance
        
        Returns:
            Complete response text
        """
        chunks = []
        async for chunk in self.generate_response(conversation_history, user_utterance):
            chunks.append(chunk)
        
        return "".join(chunks)
