(() => {
  const scriptEl = document.currentScript;
  const token = scriptEl.getAttribute("data-token");
  const bubbleTitle = scriptEl.getAttribute("data-bubble-title");
  const isLocalhost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
  const API_URL = isLocalhost ? "http://localhost:8000" : "https://knowledge-bot-retrieval.onrender.com";

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
    getSessionId() { return this.sessionId; }
  }
  const sessionManager = new SessionManager();

  let markdownLoaded = false;
  loadMarkdownLibrary().then(() => { markdownLoaded = true; });
  function loadMarkdownLibrary() {
    return new Promise((resolve) => {
      if (window.markdownit) { resolve(); return; }
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/markdown-it@14.0.0/dist/markdown-it.min.js';
      script.onload = () => {
        window.md = window.markdownit({ html: false, xhtmlOut: false, breaks: false, langPrefix: 'language-', linkify: false, typographer: false });
        resolve();
      };
      script.onerror = () => resolve();
      document.head.appendChild(script);
    });
  }

  function parseMarkdown(text) {
    if (!text) return '';
    if (window.md && markdownLoaded) {
      try {
        let processedText = text;
        processedText = processedText.replace(/([^\n])\n(### )/g, '$1\n\n$2');
        processedText = processedText.replace(/(\n- [^\n]+[.!?])\s*(\n- )/g, '$1\n$2');
        processedText = processedText.replace(/(\n- [^-\n]*[.!?])\s*- \*\*/g, '$1\n- **');
        processedText = processedText.replace(/([.!?])-\s*\*\*/g, '$1\n- **');
        processedText = processedText.replace(/([.!?])\s*([A-Z][^.!?]*[.!?])\s*-\s*\*\*/g, '$1\n\n$2\n- **');
        processedText = processedText.replace(/(\n- [^\n]+\n+)(\*\*[^*]+\*\*:)/g, '$1\n$2');
        processedText = processedText.replace(/(\*\*[^*]+\*\*:[^\n]*\n+)(\*\*[^*]+\*\*:)/g, '$1\n$2');
        processedText = processedText.replace(/([.!?])\s*(###)/g, '$1\n\n$2');
        return window.md.render(processedText);
      } catch {}
    }
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
      .replace(/\n/g, '<br>');
  }

  const STYLE = `
    :root{
      --glass: rgba(255,255,255,.12);
      --glass-strong: rgba(255,255,255,.18);
      --border: rgba(255,255,255,.35);
      --txt: #eaf0ff;
      --muted:#b7c0d6;
      --bg: #0a0c10;
      --accent:#8ea2ff;
      --neon:#7af7ff;
    }
    @media (prefers-color-scheme: light){
      :root{
        --glass: rgba(255,255,255,.55);
        --glass-strong: rgba(255,255,255,.75);
        --border: rgba(0,0,0,.10);
        --txt:#0b0f1a;
        --muted:#4b5563;
        --bg:#f5f7fb;
        --accent:#6a5cff;
        --neon:#17a2ff;
      }
    }

    .kb-btn{
      position:fixed; right:20px; bottom:20px; z-index:9999;
      display:inline-flex; align-items:center; justify-content:center;
      height:56px; width:56px; border-radius:22px; cursor:pointer; border:none;
      color:#fff; font:600 14px/1 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;
      background: radial-gradient(120% 120% at 20% 20%, var(--neon), transparent 60%),
                  linear-gradient(135deg, var(--accent), #a98bff);
      box-shadow: 0 10px 32px rgba(0,0,0,.35), 0 0 24px rgba(122,247,255,.25) inset;
      padding:0 0; white-space:nowrap;
    }
    .kb-btn.kb-has-text{ width:auto; max-width:min(60vw, 340px); padding:0 16px; border-radius:24px; }
    .kb-btn:focus{ outline:2px solid rgba(122,247,255,.6); outline-offset:2px }
    .kb-btn svg{ filter: drop-shadow(0 2px 6px rgba(0,0,0,.35)); }
    .kb-btn::after{ content:""; position:absolute; inset:0; border-radius:inherit; animation: kb-ping 2.6s infinite; pointer-events:none; box-shadow:0 0 0 0 rgba(122,247,255,.35); }
    @keyframes kb-ping { 0%{ box-shadow:0 0 0 0 rgba(122,247,255,.35) } 70%{ box-shadow:0 0 0 22px rgba(122,247,255,0) } 100%{ box-shadow:0 0 0 0 rgba(122,247,255,0) } }

    .kb-wrap{ position:fixed; right:20px; bottom:88px; z-index:9999; width: min(420px, 92vw); pointer-events:none; }
    .kb-modal{
      position:relative; border-radius:24px; background: var(--glass); border:1px solid var(--border);
      backdrop-filter: blur(22px) saturate(140%); -webkit-backdrop-filter: blur(22px) saturate(140%);
      overflow:hidden; transform: translateY(20px) scale(.98); opacity:0; pointer-events:none;
      transition: transform .25s cubic-bezier(.2,.7,.2,1), opacity .2s ease; box-shadow: 0 20px 50px rgba(0,0,0,.45);
    }
    .kb-modal.kb-open{ transform: translateY(0) scale(1); opacity:1; pointer-events:auto; }
    .kb-modal::before{ content:""; position:absolute; inset:-30% -10% auto -10%; height:60%;
      background: radial-gradient(400px 260px at 20% 40%, rgba(122,247,255,.35), transparent 60%), radial-gradient(420px 280px at 80% 0%, rgba(158,139,255,.35), transparent 65%);
      filter: blur(40px); pointer-events:none; }

    .kb-modal *, .kb-modal *::before, .kb-modal *::after { box-sizing: border-box; }
    .kb-close { position: absolute; top: 8px; right: 12px; font-size: 18px; color: var(--muted); cursor: pointer; z-index:2 }
    .kb-modal textarea { resize: none; padding: 10px; border: 1px solid var(--border); background: var(--glass-strong); color: var(--txt);
      border-radius: 12px; width: 100%; margin-top: 16px; font-family: inherit; font-size: 16px; }
    .kb-modal button { margin-top: 8px; background: linear-gradient(135deg,var(--accent),var(--neon)); color: #0a0c10; padding: 10px 12px;
      border: none; border-radius: 12px; cursor: pointer; font-size: 16px; font-weight:700 }

    .kb-answer { margin-top: 12px; background: var(--glass-strong); color: var(--txt); padding: 10px; border-radius: 12px; font-size: 14px; max-height: 56vh; overflow-y: auto; line-height: 1.5; }
    .kb-answer h1, .kb-answer h2, .kb-answer h3 { margin: 12px 0 8px 0; color: var(--txt); font-weight: 700; line-height: 1.3; }
    .kb-answer h1 { font-size: 18px; border-bottom: 2px solid var(--border); padding-bottom: 4px; }
    .kb-answer h2 { font-size: 16px; border-bottom: 1px solid var(--border); padding-bottom: 2px; }
    .kb-answer h3 { font-size: 15px; }
    .kb-answer strong { font-weight: 700; }
    .kb-answer em { font-style: italic; color: var(--muted); }
    .kb-answer ul, .kb-answer ol { margin: 8px 0; padding-left: 20px; }
    .kb-answer code { background: rgba(0,0,0,.25); padding: 2px 4px; border-radius: 3px; font-family: 'Monaco','Menlo','Ubuntu Mono',monospace; font-size: 13px; }
    .kb-answer a { color: var(--neon); text-decoration: underline; }
    .kb-answer a:hover { color: var(--accent); }
    .kb-answer p { margin: 8px 0; line-height: 1.5; }

    .kb-spinner { border: 3px solid rgba(255,255,255,.25); border-top: 3px solid var(--neon); border-radius: 50%; width: 18px; height: 18px; animation: kb-spin .8s linear infinite; display: inline-block; vertical-align: middle; }
    @keyframes kb-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    details.kb-sources { margin-top: 12px; background: rgba(142,162,255,.15); padding: 8px; border-radius: 8px; font-size: 13px; color: var(--txt) }
    summary { cursor: pointer; font-weight: 700; }

    @media (max-width:480px){ .kb-wrap{ right:12px; width: calc(100vw - 24px) } .kb-btn{ right:12px; bottom:12px } }
    @media (prefers-reduced-motion: reduce){ .kb-modal{ transition:none } .kb-btn::after{ animation:none } }
  `;

  const style = document.createElement("style");
  style.innerText = STYLE;
  document.head.appendChild(style);

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "kb-btn";
  btn.setAttribute('aria-label', 'Open AI assistant');
  btn.setAttribute('aria-expanded', 'false');
  if (bubbleTitle && bubbleTitle.trim()) {
    btn.classList.add('kb-has-text');
    btn.textContent = bubbleTitle.trim();
  } else {
    btn.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 3c5 0 9 3.6 9 8.1 0 2.4-1.3 4.5-3.3 6l.7 3.8-3.9-2c-.8.2-1.7.3-2.5.3-5 0-9-3.6-9-8.1S7 3 12 3z" fill="white" opacity=".9"/></svg>`;
  }
  document.body.appendChild(btn);

  const wrap = document.createElement('div');
  wrap.className = 'kb-wrap';
  document.body.appendChild(wrap);

  const modal = document.createElement("div");
  modal.className = "kb-modal";
  modal.setAttribute('role','dialog');
  modal.setAttribute('aria-modal','false');
  modal.innerHTML = `
    <div class="kb-close" aria-label="Close">×</div>
    <textarea rows="3" placeholder="Ask something..."></textarea>
    <button>Ask</button>
    <div class="kb-answer"></div>
  `;
  wrap.appendChild(modal);

  const textarea = modal.querySelector("textarea");
  const askBtn = modal.querySelector("button");
  const answerBox = modal.querySelector(".kb-answer");
  const closeBtn = modal.querySelector(".kb-close");

  function openPanel(){
    modal.classList.add('kb-open');
    btn.setAttribute('aria-expanded','true');
    setTimeout(() => textarea.focus(), 50);
  }
  function closePanel(){
    modal.classList.remove('kb-open');
    btn.setAttribute('aria-expanded','false');
  }

  btn.addEventListener('click', ()=>{
    const isOpen = modal.classList.contains('kb-open');
    if (isOpen) closePanel(); else openPanel();
  });

  closeBtn.addEventListener('click', ()=>{
    closePanel();
    textarea.value = '';
    answerBox.innerHTML = '';
  });

  document.addEventListener('click', (e)=>{
    const isOpen = modal.classList.contains('kb-open');
    if (!isOpen) return;
    const clickedInside = modal.contains(e.target) || btn.contains(e.target);
    if (!clickedInside) closePanel();
  });
  document.addEventListener('keydown', (e)=>{ if (e.key === 'Escape') closePanel(); });

  askBtn.onclick = async () => {
    const question = textarea.value.trim();
    if (!question) return;

    answerBox.innerHTML = `
      <div style="display: flex; align-items: center; gap: 8px; color: var(--txt)">
        <div class="kb-spinner"></div>
        <span>Thinking...</span>
      </div>
    `;
    askBtn.disabled = true;
    textarea.value = "";

    let accumulatedText = "";

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
        body: JSON.stringify({ question, session_id: sessionManager.getSessionId(), k: 12, similarity_threshold: 0.1 })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);

      const contentType = res.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        const data = await res.json();
        let raw = data.answer || data.error || "No response.";
        const sourceMatches = [...raw.matchAll(/\[source: (.+?)\]/g)];
        const uniqueSources = [...new Set(sourceMatches.map(m => m[1]))];
        raw = raw.replace(/\[source: .+?\]/g, "").trim();
        const mainHtml = parseMarkdown(raw);
        const sourcesHtml = uniqueSources.length ? `<details class="kb-sources"><summary>Show Sources (${uniqueSources.length})</summary><ul>${uniqueSources.map(src => `<li>${src}</li>`).join("")}</ul></details>` : "";
        answerBox.innerHTML = window.DOMPurify ? DOMPurify.sanitize(mainHtml + sourcesHtml) : (mainHtml + sourcesHtml);
      } else {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let responseStarted = false;
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            if (accumulatedText) {
              const sourceMatches = [...accumulatedText.matchAll(/\[source: (.+?)\]/g)];
              const uniqueSources = [...new Set(sourceMatches.map(m => m[1]))];
              const cleanText = accumulatedText.replace(/\[source: .+?\]/g, "");
              const contentText = cleanText.replace(/^Getting your response\.\.\.?\s*/, "");
              const mainHtml = parseMarkdown(contentText);
              const sourcesHtml = uniqueSources.length ? `<details class="kb-sources"><summary>Show Sources (${uniqueSources.length})</summary><ul>${uniqueSources.map(src => `<li>${src}</li>`).join("")}</ul></details>` : "";
              answerBox.innerHTML = window.DOMPurify ? DOMPurify.sanitize(mainHtml + sourcesHtml) : (mainHtml + sourcesHtml);
            }
            break;
          }
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (!line.trim()) continue;
            let dataContent = null;
            if (line.startsWith('data: ')) {
              dataContent = line.slice(6);
              if (dataContent === '[DONE]') continue;
              if (dataContent.startsWith('[ERROR]')) throw new Error(dataContent.replace('[ERROR] ', ''));
            } else {
              dataContent = line;
            }
            try {
              const jsonChunk = JSON.parse(dataContent);
              if (jsonChunk.type === 'content') accumulatedText += (jsonChunk.content || '');
              continue;
            } catch {}
            accumulatedText += dataContent === '' ? '\n' : dataContent;

            const cleanForCheck = accumulatedText.replace(/\[source: .+?\]/g, "").trim();
            const hasActualContent = cleanForCheck.length > 0 && !cleanForCheck.startsWith("Getting your response");
            if (!responseStarted && hasActualContent) responseStarted = true;

            const sourceMatches = [...accumulatedText.matchAll(/\[source: (.+?)\]/g)];
            const uniqueSources = [...new Set(sourceMatches.map(m => m[1]))];
            const cleanText = accumulatedText.replace(/\[source: .+?\]/g, "").trim();

            if (responseStarted) {
              const contentText = cleanText.replace(/^Getting your response\.\.\.?\s*/, "");
              const sourcesHtml = uniqueSources.length ? `<details class="kb-sources"><summary>Show Sources (${uniqueSources.length})</summary><ul>${uniqueSources.map(src => `<li>${src}</li>`).join("")}</ul></details>` : "";
              answerBox.innerHTML = `<pre style="white-space: pre-wrap; font-family: inherit; margin: 0; color: var(--txt)">${contentText}</pre>${sourcesHtml}`;
            } else {
              answerBox.innerHTML = `<div style="display:flex;align-items:center;gap:8px;color:var(--txt)"><div class="kb-spinner"></div><span>Thinking...</span></div>`;
            }
            answerBox.scrollTop = answerBox.scrollHeight;
          }
        }
      }
    } catch (err) {
      answerBox.innerHTML = `<strong>Error:</strong> ${err.message}`;
    } finally {
      askBtn.disabled = false;
    }
  };
})();
