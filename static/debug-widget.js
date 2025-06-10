(() => {
    const token = document.currentScript.getAttribute("data-token");
    const API_URL = "https://knowledge-bot-retrieval.onrender.com";
    
    console.log("🚀 Debug Widget Loaded");
    console.log("🔑 Token:", token ? token.substring(0, 20) + "..." : "MISSING");
    console.log("🌐 API URL:", API_URL);
  
    // Simple session ID for testing
    const sessionId = 'debug_' + Date.now();
    console.log("🔒 Session ID:", sessionId);
  
    const STYLE = `
      .debug-btn {
        position: fixed;
        bottom: 24px;
        right: 24px;
        width: 56px;
        height: 56px;
        background: #dc2626;
        border-radius: 9999px;
        color: #fff;
        font-size: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        z-index: 9999;
      }
  
      .debug-modal {
        position: fixed;
        bottom: 90px;
        right: 24px;
        width: 400px;
        max-height: 80vh;
        background: #fff;
        border: 2px solid #dc2626;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        font-family: monospace;
        z-index: 9999;
        display: none;
        flex-direction: column;
        overflow: hidden;
      }
  
      .debug-close {
        position: absolute;
        top: 8px;
        right: 12px;
        font-size: 18px;
        color: #666;
        cursor: pointer;
      }
  
      .debug-input {
        resize: none;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 8px;
        width: 100%;
        margin-top: 16px;
        font-family: inherit;
        font-size: 14px;
      }
  
      .debug-btn-ask {
        margin-top: 8px;
        background: #dc2626;
        color: #fff;
        padding: 10px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 14px;
      }
  
      .debug-output {
        margin-top: 12px;
        background: #f9fafb;
        padding: 10px;
        border-radius: 8px;
        font-size: 12px;
        max-height: 300px;
        overflow-y: auto;
        white-space: pre-wrap;
      }
    `;
  
    const style = document.createElement("style");
    style.innerText = STYLE;
    document.head.appendChild(style);
  
    const btn = document.createElement("div");
    btn.className = "debug-btn";
    btn.innerText = "🐛";
    document.body.appendChild(btn);
  
    const modal = document.createElement("div");
    modal.className = "debug-modal";
    modal.innerHTML = `
      <div class="debug-close">×</div>
      <h3 style="margin: 0; color: #dc2626;">🐛 Debug Widget</h3>
      <input class="debug-input" placeholder="Ask a question..." value="test">
      <button class="debug-btn-ask">Test Request</button>
      <div class="debug-output">Ready to test...</div>
    `;
    document.body.appendChild(modal);
  
    const textarea = modal.querySelector(".debug-input");
    const askBtn = modal.querySelector(".debug-btn-ask");
    const output = modal.querySelector(".debug-output");
    const closeBtn = modal.querySelector(".debug-close");
  
    btn.onclick = () => {
      const isOpen = modal.style.display === "flex";
      modal.style.display = isOpen ? "none" : "flex";
      if (!isOpen) {
        setTimeout(() => textarea.focus(), 50);
      }
    };
  
    closeBtn.onclick = () => {
      modal.style.display = "none";
      output.innerHTML = "Ready to test...";
    };
  
    askBtn.onclick = async () => {
      const question = textarea.value.trim();
      if (!question) return;
  
      output.innerHTML = "🚀 Starting request...\n";
      askBtn.disabled = true;
      
      const logOutput = (message) => {
        console.log(message);
        output.innerHTML += message + "\n";
        output.scrollTop = output.scrollHeight;
      };
      
      try {
        logOutput(`📤 POST ${API_URL}/ask`);
        logOutput(`📋 Question: "${question}"`);
        logOutput(`🔒 Session: ${sessionId}`);
        logOutput(`🔑 Auth: Bearer ${token ? token.substring(0, 20) + "..." : "MISSING"}`);
        
        const requestBody = {
          question,
          session_id: sessionId,
          k: 12,
          similarity_threshold: 0.1
        };
        
        logOutput(`📦 Body: ${JSON.stringify(requestBody, null, 2)}`);
        
        const res = await fetch(`${API_URL}/ask`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`
          },
          body: JSON.stringify(requestBody)
        });

        logOutput(`📥 Response Status: ${res.status} ${res.statusText}`);
        logOutput(`📋 Response Headers:`);
        for (const [key, value] of res.headers.entries()) {
          logOutput(`  ${key}: ${value}`);
        }

        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }

        const contentType = res.headers.get('content-type');
        logOutput(`🎯 Content-Type: ${contentType}`);
        
        if (contentType && contentType.includes('application/json')) {
          logOutput(`📄 Handling as JSON response`);
          const data = await res.json();
          logOutput(`📊 JSON Data: ${JSON.stringify(data, null, 2)}`);
          
        } else if (contentType && contentType.includes('text/event-stream')) {
          logOutput(`🌊 Handling as streaming response`);
          
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let chunkCount = 0;

          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              logOutput(`✅ Stream completed after ${chunkCount} chunks`);
              break;
            }

            chunkCount++;
            const chunk = decoder.decode(value, { stream: true });
            logOutput(`📦 Chunk ${chunkCount}: "${chunk}"`);
            
            const lines = chunk.split('\n');
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6);
                logOutput(`🎯 SSE Data: "${data}"`);
              }
            }
          }
        } else {
          logOutput(`❓ Unknown content type, reading as text`);
          const text = await res.text();
          logOutput(`📄 Response Text: "${text}"`);
        }
        
      } catch (err) {
        logOutput(`❌ Error: ${err.message}`);
        console.error('Debug error:', err);
      } finally {
        askBtn.disabled = false;
        logOutput(`🏁 Request completed`);
      }
    };
  })(); 