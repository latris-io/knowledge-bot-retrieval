(() => {
    const token = document.currentScript.getAttribute("data-token");
    const isLocalhost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
    const API_URL = isLocalhost ? "http://localhost:8000" : "https://knowledge-bot-retrieval.onrender.com";
  
    // Session Management
    class SessionManager {
        constructor() {
            this.sessionKey = 'kb-chat-session';
            this.sessionId = this.getOrCreateSession();
        }

        getOrCreateSession() {
            let sessionId = localStorage.getItem(this.sessionKey);
            if (!sessionId) {
                sessionId = this.generateSessionId();
                localStorage.setItem(this.sessionKey, sessionId);
            }
            return sessionId;
        }

        generateSessionId() {
            return 'widget_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }

        getSessionId() {
            return this.sessionId;
        }
    }

    const sessionManager = new SessionManager();
  
    const STYLE = `
      .kb-btn {
        position: fixed;
        bottom: 24px;
        right: 24px;
        width: 56px;
        height: 56px;
        background: #2563eb;
        border-radius: 9999px;
        color: #fff;
        font-size: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        z-index: 9999;
      }
  
      .kb-modal {
        position: fixed;
        bottom: 90px;
        right: 24px;
        width: 360px;
        max-height: 80vh;
        background: #fff;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        font-family: sans-serif;
        z-index: 9999;
        display: none;
        flex-direction: column;
        overflow: hidden;
      }
  
      .kb-modal *, .kb-modal *::before, .kb-modal *::after {
        box-sizing: border-box;
      }
  
      .kb-close {
        position: absolute;
        top: 8px;
        right: 12px;
        font-size: 18px;
        color: #666;
        cursor: pointer;
      }
  
      .kb-modal textarea {
        resize: none;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 8px;
        width: 100%;
        margin-top: 16px;
        font-family: inherit;
        font-size: 16px;
      }
  
      .kb-modal button {
        margin-top: 8px;
        background: #2563eb;
        color: #fff;
        padding: 10px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 16px;
      }
  
      .kb-answer {
        margin-top: 12px;
        background: #f9fafb;
        padding: 10px;
        border-radius: 8px;
        font-size: 14px;
        max-height: 200px;
        overflow-y: auto;
      }
  
      .kb-spinner {
        border: 3px solid #f3f3f3;
        border-top: 3px solid #2563eb;
        border-radius: 50%;
        width: 18px;
        height: 18px;
        animation: kb-spin 0.8s linear infinite;
        display: inline-block;
        vertical-align: middle;
      }
  
      @keyframes kb-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
  
      details.kb-sources {
        margin-top: 12px;
        background: #eef2ff;
        padding: 8px;
        border-radius: 6px;
        font-size: 13px;
      }
  
      summary {
        cursor: pointer;
        font-weight: bold;
      }
  
      @media (max-width: 480px) {
        .kb-modal {
          right: 12px;
          left: 12px;
          width: auto;
          bottom: 80px;
          border-radius: 12px;
          padding: 12px;
        }
        .kb-btn {
          bottom: 16px;
          right: 16px;
          width: 48px;
          height: 48px;
          font-size: 20px;
        }
      }
    `;
  
    const style = document.createElement("style");
    style.innerText = STYLE;
    document.head.appendChild(style);
  
    const btn = document.createElement("div");
    btn.className = "kb-btn";
    btn.innerText = "💬";
    document.body.appendChild(btn);
  
    const modal = document.createElement("div");
    modal.className = "kb-modal";
    modal.innerHTML = `
      <div class="kb-close">×</div>
      <textarea rows="3" placeholder="Ask something..."></textarea>
      <button>Ask</button>
      <div class="kb-answer"></div>
    `;
    document.body.appendChild(modal);
  
    const textarea = modal.querySelector("textarea");
    const askBtn = modal.querySelector("button");
    const answerBox = modal.querySelector(".kb-answer");
    const closeBtn = modal.querySelector(".kb-close");
  
    btn.onclick = () => {
      const isOpen = modal.style.display === "flex";
      modal.style.display = isOpen ? "none" : "flex";
      if (!isOpen) {
        setTimeout(() => textarea.focus(), 50);
      }
    };
  
    closeBtn.onclick = () => {
      modal.style.display = "none";
      textarea.value = "";
      answerBox.innerHTML = "";
    };
  
    askBtn.onclick = async () => {
      const question = textarea.value.trim();
      if (!question) return;
  
      answerBox.innerHTML = `
        <div style="display: flex; align-items: center; gap: 8px;">
          <div class="kb-spinner"></div>
          <span>Getting your response...</span>
        </div>
      `;
      askBtn.disabled = true;
      textarea.value = "";
      
      let accumulatedText = "";
  
      try {
        console.log(`[Widget] Making request to ${API_URL}/ask`);
        console.log(`[Widget] Question: "${question}"`);
        console.log(`[Widget] Session: ${sessionManager.getSessionId()}`);
        
        const res = await fetch(`${API_URL}/ask`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`
          },
          body: JSON.stringify({ 
            question,
            session_id: sessionManager.getSessionId(),
            k: 12,
            similarity_threshold: 0.1
          })
        });

        console.log(`[Widget] Response status: ${res.status}`);
        console.log(`[Widget] Response headers:`, Array.from(res.headers.entries()));

        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }

        // Check content type to determine response format
        const contentType = res.headers.get('content-type');
        console.log(`[Widget] Content-Type: "${contentType}"`);
        
        if (contentType && contentType.includes('application/json')) {
          console.log(`[Widget] Handling as JSON response`);
          // Handle JSON response (fallback for production)
          const data = await res.json();
          let raw = data.answer || data.error || "No response.";
          
          // Extract and deduplicate sources
          const sourceMatches = [...raw.matchAll(/\[source: (.+?)\]/g)];
          const allSources = sourceMatches.map(match => match[1]);
          const uniqueSources = [...new Set(allSources)]; // Deduplicate
          
          // Remove [source: ...] from main text
          raw = raw.replace(/\[source: .+?\]/g, "").trim();
          
          const mainHtml = marked.parse(raw);
          const sourcesHtml = uniqueSources.length
            ? `<details class="kb-sources"><summary>Show Sources (${uniqueSources.length})</summary><ul>${uniqueSources.map(src => `<li>${src}</li>`).join("")}</ul></details>`
            : "";
          
          answerBox.innerHTML = window.DOMPurify
            ? DOMPurify.sanitize(mainHtml + sourcesHtml)
            : (mainHtml + sourcesHtml);
            
        } else {
          // Handle streaming response (Server-Sent Events)
          console.log(`[Widget] Handling as streaming response`);
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let chunkCount = 0;

          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              console.log(`[Widget] Stream completed after ${chunkCount} chunks`);
              break;
            }

            chunkCount++;
            const chunk = decoder.decode(value, { stream: true });
            console.log(`[Widget] Chunk ${chunkCount}: "${chunk}"`);
            const lines = chunk.split('\n');

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6); // Remove 'data: ' prefix
                if (data === '[DONE]' || data === '') continue;
                
                if (data.startsWith('[ERROR]')) {
                  throw new Error(data.replace('[ERROR] ', ''));
                }
                
                accumulatedText += data;
                
                // Extract and deduplicate sources
                const sourceMatches = [...accumulatedText.matchAll(/\[source: (.+?)\]/g)];
                const allSources = sourceMatches.map(match => match[1]);
                const uniqueSources = [...new Set(allSources)]; // Deduplicate
                
                // Remove [source: ...] from main text for display
                const cleanText = accumulatedText.replace(/\[source: .+?\]/g, "").trim();
                
                const mainHtml = marked.parse(cleanText);
                const sourcesHtml = uniqueSources.length
                  ? `<details class="kb-sources"><summary>Show Sources (${uniqueSources.length})</summary><ul>${uniqueSources.map(src => `<li>${src}</li>`).join("")}</ul></details>`
                  : "";
                
                answerBox.innerHTML = window.DOMPurify
                  ? DOMPurify.sanitize(mainHtml + sourcesHtml)
                  : (mainHtml + sourcesHtml);
                
                // Auto-scroll to bottom of answer box
                answerBox.scrollTop = answerBox.scrollHeight;
              }
            }
          }
        }
      } catch (err) {
        console.error('[Widget] Streaming error:', err);
        answerBox.innerHTML = `<strong>Error:</strong> ${err.message}`;
      } finally {
        askBtn.disabled = false;
        console.log('[Widget] Request completed');
      }
    };
  })();
  