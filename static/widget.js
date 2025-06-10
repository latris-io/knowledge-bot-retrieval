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
  
    // Simple markdown to HTML converter
    function parseMarkdown(text) {
        if (!text) return '';
        
        // First, fix missing line breaks after headers and numbered lists
        let html = text
            // Fix header immediately followed by capital letter (start of new sentence): "### HeaderSentence" -> "### Header\nSentence"
            .replace(/^(#{1,3}\s+\w+(?:\s+\w+)*?)([A-Z]\w+)/gm, '$1\n$2')
            // Fix numbered lists: "1. ItemNext item" -> "1. Item\nNext item"  
            .replace(/^(\d+\.\s+.*?)(\d+\.)/gm, '$1\n$2')
            // Normalize paragraph breaks
            .replace(/\n\n+/g, '\n\n');
        
        // Split into lines for processing
        const lines = html.split('\n');
        const processedLines = [];
        let inList = false;
        let listType = null;
        
        for (const line of lines) {
            let processedLine = line;
            
            // Handle headers (must be processed per line to avoid conflicts)
            if (processedLine.match(/^### (.+)$/)) {
                if (inList) {
                    processedLines.push(listType === 'ol' ? '</ol>' : '</ul>');
                    inList = false;
                    listType = null;
                }
                const headerContent = processedLine.replace(/^### (.+)$/, '$1')
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/`([^`]+)`/g, '<code>$1</code>')
                    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
                processedLine = `<h3>${headerContent}</h3>`;
            } else if (processedLine.match(/^## (.+)$/)) {
                if (inList) {
                    processedLines.push(listType === 'ol' ? '</ol>' : '</ul>');
                    inList = false;
                    listType = null;
                }
                const headerContent = processedLine.replace(/^## (.+)$/, '$1')
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/`([^`]+)`/g, '<code>$1</code>')
                    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
                processedLine = `<h2>${headerContent}</h2>`;
            } else if (processedLine.match(/^# (.+)$/)) {
                if (inList) {
                    processedLines.push(listType === 'ol' ? '</ol>' : '</ul>');
                    inList = false;
                    listType = null;
                }
                const headerContent = processedLine.replace(/^# (.+)$/, '$1')
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/`([^`]+)`/g, '<code>$1</code>')
                    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
                processedLine = `<h1>${headerContent}</h1>`;
            } else if (processedLine.match(/^[\s]*[-*+]\s+/)) {
                // Handle bullet points
                if (!inList || listType !== 'ul') {
                    if (inList) {
                        processedLines.push(listType === 'ol' ? '</ol>' : '</ul>');
                    }
                    processedLines.push('<ul>');
                    inList = true;
                    listType = 'ul';
                }
                const content = processedLine.replace(/^[\s]*[-*+]\s+/, '');
                // Apply inline formatting to list content
                const formattedContent = content
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/`([^`]+)`/g, '<code>$1</code>')
                    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
                processedLine = `<li>${formattedContent}</li>`;
            } else if (processedLine.match(/^\d+\.\s+/)) {
                // Handle numbered lists
                if (!inList || listType !== 'ol') {
                    if (inList) {
                        processedLines.push(listType === 'ol' ? '</ol>' : '</ul>');
                    }
                    processedLines.push('<ol>');
                    inList = true;
                    listType = 'ol';
                }
                const content = processedLine.replace(/^\d+\.\s+/, '');
                // Apply inline formatting to list content
                const formattedContent = content
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/`([^`]+)`/g, '<code>$1</code>')
                    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
                processedLine = `<li>${formattedContent}</li>`;
            } else {
                // Regular content
                if (inList) {
                    processedLines.push(listType === 'ol' ? '</ol>' : '</ul>');
                    inList = false;
                    listType = null;
                }
                // Only add non-empty lines or preserve intentional empty lines
                if (processedLine.trim() || processedLines.length > 0) {
                    // Apply inline formatting
                    processedLine = processedLine
                        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // Bold
                        .replace(/\*(.*?)\*/g, '<em>$1</em>')              // Italic
                        .replace(/`([^`]+)`/g, '<code>$1</code>')          // Code
                        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>'); // Links
                }
            }
            
            // Add the processed line
            if (processedLine.trim() || processedLines.length > 0) {
                processedLines.push(processedLine);
            }
        }
        
        // Close any open list
        if (inList) {
            processedLines.push(listType === 'ol' ? '</ol>' : '</ul>');
        }
        
        // Join lines and handle paragraph breaks
        html = processedLines.join('\n');
        
        // Convert double newlines to paragraph breaks, single newlines to line breaks
        html = html.replace(/\n\n+/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');
        
        // Wrap in paragraph tags if there's content that needs it
        if (html && !html.match(/^<(h[1-6]|ul|ol|li)/)) {
            html = '<p>' + html + '</p>';
        }
        
        // Clean up empty paragraphs and extra breaks
        html = html.replace(/<p><\/p>/g, '');
        html = html.replace(/<p><br><\/p>/g, '');
        html = html.replace(/(<br>){3,}/g, '<br><br>');
        
        return html.trim();
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
  