#!/usr/bin/env python3
"""
Streaming Diagnostic Tool for Knowledge Bot Retrieval Service
Identifies and diagnoses streaming response issues
"""

import asyncio
import json
import aiohttp
import time
from typing import AsyncGenerator

class StreamingDiagnostic:
    def __init__(self, base_url: str = "https://knowledge-bot-retrieval.onrender.com"):
        self.base_url = base_url
        self.token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb21wYW55X2lkIjozLCJib3RfaWQiOjF9.ytHVcMRM99aAkFMg_U1I4VZbz3mYxskzzxSUORe3ico"
        
    async def test_streaming_response(self, question: str = "What industries are represented?"):
        """Test the streaming response and diagnose issues"""
        print(f"🧪 STREAMING DIAGNOSTIC TEST")
        print(f"📍 URL: {self.base_url}/ask")
        print(f"❓ Question: {question}")
        print("=" * 60)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        
        payload = {
            "question": question,
            "session_id": "diagnostic_test",
            "k": 8,
            "similarity_threshold": 0.1
        }
        
        start_time = time.time()
        chunk_count = 0
        content_received = False
        
        try:
            async with aiohttp.ClientSession() as session:
                print(f"🚀 Making request...")
                async with session.post(f"{self.base_url}/ask", headers=headers, json=payload) as response:
                    print(f"📥 Response Status: {response.status}")
                    print(f"📋 Response Headers:")
                    for key, value in response.headers.items():
                        print(f"   {key}: {value}")
                    print()
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '')
                    if 'text/event-stream' not in content_type:
                        print(f"❌ PROBLEM: Expected 'text/event-stream', got '{content_type}'")
                        text_response = await response.text()
                        print(f"📄 Response Body: {text_response[:500]}...")
                        return
                    
                    print(f"✅ Correct Content-Type: {content_type}")
                    print(f"🌊 Processing streaming chunks...")
                    print("-" * 40)
                    
                    # Process streaming chunks
                    async for chunk in self._read_sse_stream(response):
                        chunk_count += 1
                        elapsed = time.time() - start_time
                        
                        if chunk_count <= 10 or chunk_count % 5 == 0:  # Show first 10, then every 5th
                            print(f"📦 Chunk {chunk_count} (t={elapsed:.2f}s): {chunk[:100]}...")
                        
                        if not content_received and chunk.strip() and 'start' not in chunk.lower():
                            content_received = True
                            print(f"🎯 First content received at t={elapsed:.2f}s")
                        
                        # Check if it's JSON structured data
                        if chunk.startswith('data: {'):
                            try:
                                json_data = json.loads(chunk[6:])  # Remove 'data: ' prefix
                                if json_data.get('type') == 'content':
                                    print(f"📝 Content chunk: '{json_data.get('content', '')[:50]}...'")
                            except json.JSONDecodeError:
                                pass
                                
        except Exception as e:
            print(f"💥 ERROR: {e}")
            return
            
        total_time = time.time() - start_time
        print("-" * 40)
        print(f"✅ DIAGNOSTIC COMPLETE")
        print(f"📊 Total chunks: {chunk_count}")
        print(f"⏱️ Total time: {total_time:.2f}s")
        print(f"🎯 Content received: {content_received}")
        
        if chunk_count == 0:
            print("❌ MAJOR ISSUE: No chunks received!")
        elif not content_received:
            print("⚠️ ISSUE: Chunks received but no actual content detected")
        elif chunk_count < 5:
            print("⚠️ ISSUE: Very few chunks - streaming may not be working properly")
        else:
            print("✅ Streaming appears to be working!")
            
    async def _read_sse_stream(self, response) -> AsyncGenerator[str, None]:
        """Read Server-Sent Events stream"""
        buffer = ""
        async for chunk in response.content.iter_chunked(1024):
            if chunk:
                buffer += chunk.decode('utf-8')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        yield line.strip()
                        
    async def test_non_streaming_fallback(self, question: str = "What industries are represented?"):
        """Test if the system falls back to non-streaming"""
        print(f"\n🔄 TESTING NON-STREAMING FALLBACK")
        print("=" * 60)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        
        payload = {
            "question": question,
            "session_id": "diagnostic_fallback",
            "k": 8,
            "similarity_threshold": 0.1
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/ask", headers=headers, json=payload) as response:
                    content_type = response.headers.get('content-type', '')
                    print(f"📋 Content-Type: {content_type}")
                    
                    if 'application/json' in content_type:
                        print("📄 Received JSON response (non-streaming fallback)")
                        data = await response.json()
                        print(f"📝 Response keys: {list(data.keys())}")
                        if 'result' in data:
                            print(f"✅ Has 'result' field: {len(data['result'])} chars")
                        if 'source_documents' in data:
                            print(f"✅ Has sources: {len(data['source_documents'])} documents")
                    else:
                        print(f"❌ Unexpected content type: {content_type}")
                        
        except Exception as e:
            print(f"💥 ERROR: {e}")

async def main():
    diagnostic = StreamingDiagnostic()
    
    print("🔬 KNOWLEDGE BOT STREAMING DIAGNOSTICS")
    print("=" * 60)
    
    # Test streaming
    await diagnostic.test_streaming_response()
    
    # Test non-streaming fallback
    await diagnostic.test_non_streaming_fallback()
    
    print("\n💡 RECOMMENDATIONS:")
    print("1. Check server logs for streaming handler errors")
    print("2. Verify LangChain callbacks are working correctly") 
    print("3. Test with different query types and lengths")
    print("4. Check if browser client logic handles chunks properly")

if __name__ == "__main__":
    asyncio.run(main()) 