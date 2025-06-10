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
    
    // Initialize marked.js library
    loadMarkedLibrary();
  
    // Load marked.js library for proper markdown parsing
    function loadMarkedLibrary() {
        return new Promise((resolve) => {
            if (window.marked) {
                resolve();
                return;
            }
            
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
            script.onload = () => {
                // Configure marked.js for LLM output
                marked.setOptions({
                    breaks: true,        // Convert \n to <br>
                    gfm: true,           // GitHub Flavored Markdown
                    sanitize: false,     // We'll handle sanitization elsewhere
                    smartypants: false   // Keep simple quotes
                });
                resolve();
            };
            document.head.appendChild(script);
        });
    }

    // Standard markdown parsing using marked.js with LLM preprocessing
    function parseMarkdown(text) {
        if (!text) return '';
        
        // Pre-process common LLM markdown issues
        let processed = text
            // Fix orphaned header markers at end of lines
            .replace(/#{4,6}\s*$/gm, '')
            // Fix headers running into subheaders: "### Header#### Sub" -> "### Header\n#### Sub"
            .replace(/(#{1,6}\s+[^#\n]*?)(#{1,6}\s+)/gm, '$1\n$2')
            // Fix headers running into numbered lists: "### Header1. Item" -> "### Header\n1. Item"
            .replace(/(#{1,6}\s+[^#\n]*?)(\d+\.\s+)/gm, '$1\n$2')
            // Fix numbered lists running together: "1. Item2. Next" -> "1. Item\n2. Next"
            .replace(/(\d+\.\s+[^\n]*?)(\d+\.\s+)/gm, '$1\n$2')
            // Fix bullet points running together: "- Item- Next" -> "- Item\n- Next"
            .replace(/(-\s+[^\n]*?)(-\s+)/gm, '$1\n$2')
            // Fix time ranges being interpreted as lists: "- 7:00 AM - 4:00 PM" -> "7:00 AM - 4:00 PM"
            .replace(/^(\s*)-\s+(\d+:\d+\s+[AP]M\s+-\s+\d+:\d+\s+[AP]M)/gm, '$1$2')
            // Clean up trailing header markers
            .replace(/#{1,6}\s*$/, '');

        // Use marked.js if available, fallback to simple parsing
        if (window.marked) {
            try {
                return marked.parse(processed);
            } catch (error) {
                console.warn('Marked.js parsing failed, using fallback:', error);
            }
        }
        
        // Fallback: basic inline formatting only
        return processed
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
            .replace(/\n/g, '<br>');
    }
  
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
        line-height: 1.5;
      }

      .kb-answer h1, .kb-answer h2, .kb-answer h3 {
        margin: 12px 0 8px 0;
        color: #1f2937;
        font-weight: 600;
        line-height: 1.3;
      }

      .kb-answer h1 {
        font-size: 18px;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 4px;
      }

      .kb-answer h2 {
        font-size: 16px;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 2px;
      }

      .kb-answer h3 {
        font-size: 15px;
      }

      .kb-answer strong {
        font-weight: 600;
        color: #1f2937;
      }

      .kb-answer em {
        font-style: italic;
        color: #4b5563;
      }

      .kb-answer ul {
        margin: 8px 0;
        padding-left: 20px;
        list-style-type: disc;
      }

      .kb-answer ol {
        margin: 8px 0;
        padding-left: 20px;
        list-style-type: decimal;
      }

      .kb-answer li {
        margin: 4px 0;
        line-height: 1.4;
      }

      .kb-answer code {
        background: #e5e7eb;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 13px;
        color: #1f2937;
      }

      .kb-answer a {
        color: #2563eb;
        text-decoration: underline;
      }

      .kb-answer a:hover {
        color: #1d4ed8;
      }

      .kb-answer br {
        line-height: 1.5;
      }

      .kb-answer p {
        margin: 8px 0;
        line-height: 1.5;
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
          <span>Thinking...</span>
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
          
          // Parse markdown to HTML
          const mainHtml = parseMarkdown(raw);
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
          let responseStarted = false; // Track if actual response content has started

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
                
                // Check if we have actual response content (not just loading message)
                const cleanForCheck = accumulatedText.replace(/\[source: .+?\]/g, "").trim();
                const hasActualContent = cleanForCheck.length > 0 && !cleanForCheck.startsWith("Getting your response");
                
                if (!responseStarted && hasActualContent) {
                  responseStarted = true;
                  console.log(`[Widget] Response content detected, hiding Thinking spinner`);
                }
                
                // Extract and deduplicate sources
                const sourceMatches = [...accumulatedText.matchAll(/\[source: (.+?)\]/g)];
                const allSources = sourceMatches.map(match => match[1]);
                const uniqueSources = [...new Set(allSources)]; // Deduplicate
                
                // Remove [source: ...] from main text for display
                const cleanText = accumulatedText.replace(/\[source: .+?\]/g, "").trim();
                
                if (responseStarted) {
                  // Show actual content, removing any loading messages
                  const contentText = cleanText.replace(/^Getting your response\.\.\.?\s*/, "").trim();
                  
                  // Parse markdown to HTML
                  const mainHtml = parseMarkdown(contentText);
                  const sourcesHtml = uniqueSources.length
                    ? `<details class="kb-sources"><summary>Show Sources (${uniqueSources.length})</summary><ul>${uniqueSources.map(src => `<li>${src}</li>`).join("")}</ul></details>`
                    : "";
                  
                  answerBox.innerHTML = window.DOMPurify
                    ? DOMPurify.sanitize(mainHtml + sourcesHtml)
                    : (mainHtml + sourcesHtml);
                } else {
                  // Still in loading phase, show Thinking with spinner
                  answerBox.innerHTML = `
                    <div style="display: flex; align-items: center; gap: 8px;">
                      <div class="kb-spinner"></div>
                      <span>Thinking...</span>
                    </div>
                  `;
                }
                
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
  