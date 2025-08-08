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
    .kb-close { margin-left:6px; inline-size:28px; block-size:28px; display:grid; place-items:center; border-radius:8px; color: var(--muted); cursor: pointer; z-index:2 }

    /* Header */
    .kb-head{ position:relative; z-index:2; display:flex; align-items:center; gap:12px; padding:12px 14px; border-bottom:1px solid rgba(255,255,255,.08); background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02)) }
    .kb-logo{ inline-size:28px; block-size:28px; border-radius:10px; background: linear-gradient(135deg, var(--accent), var(--neon)); box-shadow: 0 0 16px rgba(122,247,255,.35) }
    .kb-title{ font-weight:700; letter-spacing:.2px; color: var(--txt) }
    .kb-badges{ margin-left:auto; display:flex; gap:8px }
    .kb-pill{ font-size:12px; color:var(--muted); padding:4px 8px; border-radius:999px; background: var(--glass-strong); border:1px solid var(--border) }
    .kb-close:hover{ background:rgba(255,255,255,.10) }

    /* Messages */
    .kb-msgs{ max-height:min(68vh, 560px); overflow:auto; padding:14px; display:flex; flex-direction:column; gap:10px }
    .kb-msg{ display:flex; gap:10px; max-width:78% }
    .kb-msg.user{ align-self:flex-end; flex-direction:row-reverse }
    .kb-ava{ width:28px; height:28px; border-radius:10px; flex:0 0 auto; background:#fff }
    .kb-msg.ai .kb-ava{ background: linear-gradient(135deg, var(--accent), var(--neon)) }
    .kb-bubble{ padding:10px 12px; border-radius:14px; background: var(--glass-strong); border:1px solid var(--border); box-shadow:0 10px 24px rgba(0,0,0,.18); color: var(--txt) }
    .kb-msg.user .kb-bubble{ background: linear-gradient(135deg, rgba(155,140,255,.35), rgba(135,245,255,.30)); border-color: rgba(255,255,255,.55) }
    .kb-meta{ font-size:11px; color:var(--muted); margin-top:6px }

    /* Dock */
    .kb-dock{ display:grid; grid-template-columns:auto 1fr auto; gap:10px; align-items:center; padding:10px; border-top:1px solid rgba(255,255,255,.06); background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02)) }
    .kb-tools{ display:flex; gap:6px; padding:6px; border-radius:12px; background:var(--glass-strong); border:1px solid var(--border) }
    .kb-tools button{ all:unset; cursor:pointer; padding:6px 8px; border-radius:10px; color: var(--txt) }
    .kb-tools button:hover{ background:rgba(255,255,255,.12) }
    .kb-input{ display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:14px; background:var(--glass-strong); border:1px solid var(--border) }
    .kb-input input{ all:unset; flex:1; color: var(--txt) }
    .kb-send{ all:unset; cursor:pointer; padding:8px 12px; border-radius:12px; font-weight:700; background:linear-gradient(135deg,var(--accent),var(--neon)); box-shadow:0 8px 24px rgba(23,162,255,.35); color:#0a0c10 }

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
    <div class="kb-head" role="banner">
      <div class="kb-logo" aria-hidden="true"></div>
      <div class="kb-title">Nova AI Concierge</div>
      <div class="kb-badges">
        <span class="kb-pill">Private</span>
        <span class="kb-pill">Multimodal</span>
      </div>
      <button class="kb-close" title="Close" aria-label="Close">✕</button>
    </div>

    <div class="kb-msgs" id="kbMsgs"></div>

    <div class="kb-dock">
      <div class="kb-tools" aria-label="input tools">
        <button title="Voice">��</button>
        <button title="Camera">📷</button>
        <button title="Upload">📎</button>
      </div>
      <label class="kb-input" aria-label="chat input">
        <input id="kbInput" placeholder="Ask anything… 'Plan a 2‑day coastal escape with oysters + live jazz'" />
      </label>
      <button class="kb-send" id="kbSend">Send</button>
    </div>
  `;
  wrap.appendChild(modal);
  
  const msgsEl = modal.querySelector('#kbMsgs');
  const inputEl = modal.querySelector('#kbInput');
  const sendEl  = modal.querySelector('#kbSend');
  const closeBtn = modal.querySelector('.kb-close');
  
  function openPanel(){
    modal.classList.add('kb-open');
    btn.setAttribute('aria-expanded','true');
    setTimeout(() => inputEl?.focus(), 120);
  }
  function closePanel(){
    modal.classList.remove('kb-open');
    btn.setAttribute('aria-expanded','false');
  }

  btn.addEventListener('click', ()=>{
    const isOpen = modal.classList.contains('kb-open');
    if (isOpen) closePanel(); else openPanel();
  });

  closeBtn.addEventListener('click', ()=>{ closePanel(); inputEl.value = ''; });

  document.addEventListener('click', (e)=>{
    const isOpen = modal.classList.contains('kb-open');
    if (!isOpen) return;
    const clickedInside = modal.contains(e.target) || btn.contains(e.target);
    if (!clickedInside) closePanel();
  });
  document.addEventListener('keydown', (e)=>{ if (e.key === 'Escape') closePanel(); });

  function addMsg(role, html){
    const node = document.createElement('div');
    node.className = 'kb-msg ' + role;
    node.innerHTML = `
      <div class="kb-ava" ${role==='user'? 'style="background:#fff"' : ''}></div>
      <div><div class="kb-bubble">${html}</div><div class="kb-meta">${role==='user'?'You':'Nova'} • now</div></div>`;
    msgsEl.appendChild(node);
    msgsEl.scrollTop = msgsEl.scrollHeight;
    return node.querySelector('.kb-bubble');
  }

  async function askQuestion(q){
    if(!q) return;
    const userBubble = addMsg('user', (window.DOMPurify?DOMPurify.sanitize(q):q));
    const aiBubble = addMsg('ai', `<div style="display:flex;align-items:center;gap:8px"><span class="kb-spinner"></span><span>Thinking…</span></div>`);

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ question: q, session_id: sessionManager.getSessionId(), k: 12, similarity_threshold: 0.1 })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);

      const contentType = res.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        const data = await res.json();
        let raw = (data.answer || data.error || 'No response.').trim();
        const sourceMatches = [...raw.matchAll(/\[source: (.+?)\]/g)];
        const uniqueSources = [...new Set(sourceMatches.map(m => m[1]))];
        raw = raw.replace(/\[source: .+?\]/g, '');
        const mainHtml = parseMarkdown(raw);
        const sourcesHtml = uniqueSources.length ? `<details class="kb-sources"><summary>Show Sources (${uniqueSources.length})</summary><ul>${uniqueSources.map(s=>`<li>${s}</li>`).join('')}</ul></details>` : '';
        aiBubble.innerHTML = (window.DOMPurify ? DOMPurify.sanitize(mainHtml + sourcesHtml) : (mainHtml + sourcesHtml));
        msgsEl.scrollTop = msgsEl.scrollHeight;
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let acc = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        for (const line of chunk.split('\n')) {
          if (!line.trim()) continue;
          let dataContent = line.startsWith('data: ')? line.slice(6) : line;
          if (dataContent === '[DONE]') continue;
          if (dataContent.startsWith('[ERROR]')) throw new Error(dataContent.replace('[ERROR] ', ''));
          try { const json = JSON.parse(dataContent); if (json.type === 'content') acc += (json.content || ''); else acc += dataContent; }
          catch { acc += dataContent; }
          const clean = acc.replace(/\[source: .+?\]/g, '').replace(/^Getting your response\.\.\.?\s*/, '');
          aiBubble.innerHTML = `<div style="white-space:pre-wrap">${clean}</div>`;
          msgsEl.scrollTop = msgsEl.scrollHeight;
        }
      }

      const srcs = [...acc.matchAll(/\[source: (.+?)\]/g)].map(m=>m[1]);
      const unique = [...new Set(srcs)];
      const cleanFinal = acc.replace(/\[source: .+?\]/g, '').replace(/^Getting your response\.\.\.?\s*/, '');
      const mainHtml = parseMarkdown(cleanFinal);
      const sourcesHtml = unique.length ? `<details class="kb-sources"><summary>Show Sources (${unique.length})</summary><ul>${unique.map(s=>`<li>${s}</li>`).join('')}</ul></details>` : '';
      aiBubble.innerHTML = (window.DOMPurify ? DOMPurify.sanitize(mainHtml + sourcesHtml) : (mainHtml + sourcesHtml));
    } catch (e) {
      aiBubble.innerHTML = `<strong>Error:</strong> ${e.message}`;
    }
  }

  sendEl.addEventListener('click', ()=>{ const q = (inputEl.value||'').trim(); if(!q) return; inputEl.value=''; askQuestion(q); });
  inputEl.addEventListener('keydown', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); const q=(inputEl.value||'').trim(); if(q){ inputEl.value=''; askQuestion(q); } } });
  })();
