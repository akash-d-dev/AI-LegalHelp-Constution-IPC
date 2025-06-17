#!/usr/bin/env python3
"""
Test script for Constitutional AI Chat API
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.services.chat_service import chat_service
from app.models.chat import ChatMessage

async def test_chat_service():
    """Test the chat service functionality"""
    
    print("🧪 Testing Constitutional AI Chat Service")
    print("=" * 50)
    
    # Test 1: Basic message processing
    print("\n📝 Test 1: Basic message processing")
    try:
        response = await chat_service.process_message(
            message="What are fundamental rights?",
            chat_history=[]
        )
        print(f"✅ Response received: {response.message[:100]}...")
        print(f"   Processing time: {response.processing_time:.2f}s")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Message with chat history
    print("\n💬 Test 2: Message with chat history")
    try:
        chat_history = [
            ChatMessage(
                content="Hello",
                sender="user",
                timestamp="2024-01-01T12:00:00Z"
            ),
            ChatMessage(
                content="Hello! How can I help you with Indian law today?",
                sender="agent", 
                timestamp="2024-01-01T12:00:01Z"
            )
        ]
        
        response = await chat_service.process_message(
            message="Tell me about Article 21",
            chat_history=chat_history
        )
        print(f"✅ Response with context: {response.message[:100]}...")
        print(f"   Processing time: {response.processing_time:.2f}s")
        print(f"   History length processed: {response.metadata.get('agent_history_length', 0)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Health check
    print("\n🏥 Test 3: Health check")
    try:
        health = await chat_service.health_check()
        print(f"✅ Health status: {health.get('agent_status', 'unknown')}")
        print(f"   Test successful: {health.get('test_successful', False)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Chat service testing completed!")

if __name__ == "__main__":
    asyncio.run(test_chat_service()) 