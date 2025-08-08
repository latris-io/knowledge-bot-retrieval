(() => {
  const scriptEl = document.currentScript;
  const token = scriptEl.getAttribute("data-token");
  const bubbleTitle = scriptEl.getAttribute("data-bubble-title");
  const searchTitle = scriptEl.getAttribute("data-search-title") || "";
  const showLogo = (scriptEl.getAttribute("data-show-logo") === "true");
    const isLocalhost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
    const API_URL = isLocalhost ? "http://localhost:8000" : "https://knowledge-bot-retrieval.onrender.com";
  
    // Host element + Shadow DOM to isolate styles from the host site
    const host = document.createElement('div');
    host.id = 'kb-widget-host';
    // Keep host out of layout flow; children use position:fixed
    host.style.all = 'initial';
    host.style.position = 'relative';
    host.style.zIndex = '2147483646';
    document.body.appendChild(host);
    const shadowRoot = host.attachShadow ? host.attachShadow({ mode: 'open' }) : null;

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
      --glass: rgba(255,255,255,.18);
      --glass-strong: rgba(255,255,255,.34);
      --border: rgba(255,255,255,.35);
      --txt: #eaf0ff;
      --muted:#b7c0d6;
      --bg: #0a0c10;
      --accent:#8ea2ff;
      --neon:#7af7ff;
      /* Shared brand gradients */
      --brand-gradient: linear-gradient(135deg, rgba(155,140,255,.35), rgba(135,245,255,.30));
      --brand-overlay: linear-gradient(180deg, rgba(24,28,38,.78), rgba(24,28,38,.78));
      /* Bubble themes */
      --assistant-bubble-bg: linear-gradient(180deg, rgba(255,255,255,1.0), rgba(245,248,255,.98));
      --assistant-text: #0b0f1a;
      --user-bubble-gradient: linear-gradient(135deg, #aee6ff 0%, #c9b8ff 35%, #ffd6e7 70%, #fff3b8 100%);
      --user-bubble-overlay: linear-gradient(180deg, rgba(255,255,255,.22), rgba(255,255,255,.10));
      --user-bubble-border: rgba(255,255,255,.65);
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
        --assistant-text:#0b0f1a;
      }
    }

    /* Enforce font across the entire widget scope */
    :host, .kb-modal, .kb-modal * { font-family: -apple-system,BlinkMacSystemFont,sans-serif !important; }

    .kb-btn{
      position:fixed; right:20px; bottom:20px; z-index:9999;
      display:inline-flex; align-items:center; justify-content:center;
      height:56px; width:56px; border-radius:22px; cursor:pointer; border:none;
      color:#fff; font:600 14px/1 -apple-system,BlinkMacSystemFont,sans-serif;
      background: var(--brand-gradient);
      box-shadow: 0 10px 32px rgba(0,0,0,.35), 0 0 24px rgba(122,247,255,.25) inset;
      padding:0 0; white-space:nowrap;
      }
    .kb-btn.kb-has-text{ width:auto; max-width:min(60vw, 340px); padding:0 16px; border-radius:24px; }
    .kb-btn:focus{ outline:2px solid rgba(122,247,255,.6); outline-offset:2px }
    .kb-btn svg{ filter: drop-shadow(0 2px 6px rgba(0,0,0,.35)); }
    .kb-btn::after{ content:""; position:absolute; inset:0; border-radius:inherit; animation: kb-ping 2.6s infinite; pointer-events:none; box-shadow:0 0 0 0 rgba(122,247,255,.35); }
    @keyframes kb-ping { 0%{ box-shadow:0 0 0 0 rgba(122,247,255,.35) } 70%{ box-shadow:0 0 0 22px rgba(122,247,255,0) } 100%{ box-shadow:0 0 0 0 rgba(122,247,255,0) } }

    .kb-wrap{ position:fixed; right:20px; bottom:88px; z-index:9999; width: min(420px, 92vw); max-height:100vh; pointer-events:none; }
    .kb-modal{
      position:fixed; right:20px; bottom:88px; width:min(420px, 92vw); border-radius:24px;
      /* internal fog tint over glass to avoid page punch-through */
      background: linear-gradient(180deg, rgba(10,12,16,.42), rgba(10,12,16,.28)), var(--glass);
      border:1px solid var(--border);
      backdrop-filter: blur(22px) saturate(140%); -webkit-backdrop-filter: blur(22px) saturate(140%);
      overflow:hidden; transform: translateY(20px) scale(.98); opacity:0; pointer-events:none;
      max-height: min(90vh, calc(100vh - 24px));
      display:flex; flex-direction:column;
      transition: transform .25s cubic-bezier(.2,.7,.2,1), opacity .2s ease; box-shadow: 0 20px 50px rgba(0,0,0,.45);
    }
    .kb-modal.kb-open{ transform: translateY(0) scale(1); opacity:1; pointer-events:auto; }
    @supports (height: 100dvh) {
      .kb-modal{ max-height: min(90dvh, calc(100dvh - 24px)); }
    }
    .kb-modal::before{ content:""; position:absolute; inset:-30% -10% auto -10%; height:60%;
      background: radial-gradient(400px 260px at 20% 40%, rgba(122,247,255,.18), transparent 60%), radial-gradient(420px 280px at 80% 0%, rgba(158,139,255,.18), transparent 65%);
      filter: blur(40px); pointer-events:none; }
    .kb-modal::after{ content:""; position:absolute; inset:0; border-radius:inherit; pointer-events:none;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.25), inset 0 0 0 1px rgba(255,255,255,.06); }

    .kb-modal *, .kb-modal *::before, .kb-modal *::after { box-sizing: border-box; }
    .kb-close { position:absolute; top:8px; right:10px; inline-size:32px; block-size:32px; display:grid; place-items:center; border-radius:10px; color: var(--muted); cursor: pointer; z-index:3; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); }
    .kb-close svg { width:16px; height:16px; }
     
    /* Header */
    .kb-head{ position:sticky; top:0; z-index:2; display:flex; align-items:center; gap:12px; padding:10px 12px; border-bottom:1px solid rgba(255,255,255,.08); background: linear-gradient(180deg, rgba(255,255,255,.10), rgba(255,255,255,.02)) }
    .kb-logo{ inline-size:22px; block-size:22px; border-radius:8px; background: linear-gradient(135deg, var(--accent), var(--neon)); box-shadow: 0 0 10px rgba(122,247,255,.35) }
    .kb-title{ font-weight:700; letter-spacing:.2px; color: var(--txt); font-size:14px }
    .kb-title{ margin-left:4px; font-family: -apple-system,BlinkMacSystemFont,sans-serif; }
    .kb-badges{ display:none }
    .kb-pill{ display:none }
    .kb-close:hover{ background:rgba(255,255,255,.14) }

    /* Messages */
    .kb-msgs{ flex:1 1 auto; overflow-y:auto; overflow-x:hidden; padding:14px; display:flex; flex-direction:column; gap:10px; overscroll-behavior: contain; -webkit-overflow-scrolling: touch; background: linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.03)) }
    .kb-msg{ display:flex; gap:0; max-width:86% }
    .kb-msg.user{ align-self:flex-end; flex-direction:row-reverse }
    .kb-ava{ display:none }
    .kb-msg.ai .kb-ava{ display:none }
             /* Assistant bubble (light glass, darker contrast, no blur on bubble) */
      .kb-bubble{ padding:12px 14px; border-radius:18px; background: var(--assistant-bubble-bg); border:1px solid rgba(0,0,0,.20); box-shadow:0 16px 40px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.65); color: var(--assistant-text); font-size:13px; line-height:1.5; word-break: break-word; overflow-wrap: anywhere; white-space: pre-wrap; max-width: 100%; backdrop-filter:none !important; -webkit-backdrop-filter:none !important; }
    .kb-bubble *{ max-width:100%; box-sizing:border-box }
    .kb-bubble pre, .kb-bubble code{ white-space: pre-wrap; word-break: break-word; overflow-wrap:anywhere }
    .kb-bubble table{ display:block; width:100%; overflow:auto }
    .kb-bubble h1, .kb-bubble h2, .kb-bubble h3 { font-size:15px; }
    /* User bubble (pastel gradient over opaque solid, white text) */
    .kb-msg.user .kb-bubble{ background: linear-gradient(180deg, rgba(0,0,0,.16), rgba(0,0,0,.10)), linear-gradient(135deg, rgba(155,140,255,.80), rgba(135,245,255,.70)); color:#fff; border-color: rgba(255,255,255,.85); box-shadow:0 16px 42px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.62); backdrop-filter:none; -webkit-backdrop-filter:none; opacity:1; background-clip: padding-box; }
    .kb-meta{ font-size:11px; color:var(--muted); margin-top:6px }

    /* Dock */
    .kb-dock{ display:grid; grid-template-columns: 1fr auto; gap:10px; align-items:center; padding:12px; border-top:1px solid rgba(255,255,255,.06); background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02)) }
    .kb-input{ display:flex; align-items:center; gap:10px; padding:10px 12px; border-radius:14px; background:var(--glass-strong); border:1px solid var(--border) }
    .kb-input input{ all:unset; flex:1; color: var(--txt); font-family: -apple-system,BlinkMacSystemFont,sans-serif; font-size:13px !important; line-height:1.5; font-weight:400 }
    .kb-input input::placeholder{ color: var(--muted); opacity:.9 }
    .kb-send{ all:unset; cursor:pointer; padding:10px 18px; border-radius:16px; font-weight:700; color:#fff;
      background: var(--brand-gradient);
      border:1px solid rgba(255,255,255,.40);
      box-shadow: 0 10px 30px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.85), 0 0 28px rgba(122,247,255,.22);
      backdrop-filter: blur(12px) saturate(170%);
      -webkit-backdrop-filter: blur(12px) saturate(170%);
      text-shadow: 0 1px 2px rgba(0,0,0,.30);
    }
    .kb-send:hover{ filter: brightness(1.05); }
    .kb-send:active{ transform: translateY(1px); }

    .kb-spinner { border: 3px solid rgba(255,255,255,.25); border-top: 3px solid var(--neon); border-radius: 50%; width: 18px; height: 18px; animation: kb-spin .8s linear infinite; display: inline-block; vertical-align: middle; }
    @keyframes kb-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    details.kb-sources { margin-top: 12px; background: rgba(142,162,255,.12); padding: 10px; border-radius: 10px; font-size: 13px; color: var(--txt); border:1px solid rgba(255,255,255,.18) }
    details.kb-sources summary { cursor: pointer; font-weight:700; color: var(--neon); text-decoration: underline; }
    details.kb-sources ul{ margin:8px 0 0 18px; }
  
    @media (max-width:480px){
      .kb-wrap{ right:12px; width: calc(100vw - 24px); max-height:100vh }
      .kb-modal{ right:12px; bottom:72px; width: calc(100vw - 24px); max-height: calc(100vh - 96px); }
      @supports (height: 100dvh) {
        .kb-modal{ max-height: calc(100dvh - 96px); }
      }
      .kb-btn{ right:12px; bottom:12px }
      /* Prevent iOS zoom-on-focus by ensuring input font-size >= 16px */
      .kb-input input{ font-size:16px !important }
    }
    @media (prefers-reduced-motion: reduce){ .kb-modal{ transition:none } .kb-btn::after{ animation:none } }
    `;
  
    const style = document.createElement("style");
    style.textContent = STYLE;
    if (shadowRoot) { shadowRoot.appendChild(style); } else { document.head.appendChild(style); }
  
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
    if (shadowRoot) { shadowRoot.appendChild(btn); } else { document.body.appendChild(btn); }

  const wrap = document.createElement('div');
  wrap.className = 'kb-wrap';
  if (shadowRoot) { shadowRoot.appendChild(wrap); } else { document.body.appendChild(wrap); }
  
    const modal = document.createElement("div");
    modal.className = "kb-modal";
  modal.setAttribute('role','dialog');
  modal.setAttribute('aria-modal','false');
    modal.innerHTML = `
    <div class="kb-head" role="banner">
      <div class="kb-logo" aria-hidden="true"></div>
      <div class="kb-title">${searchTitle}</div>
      <button class="kb-close" title="Close" aria-label="Close"><svg viewBox="0 0 24 24" fill="none"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></button>
    </div>

    <div class="kb-msgs" id="kbMsgs"></div>

    <div class="kb-dock">
      <label class="kb-input" aria-label="chat input">
        <input id="kbInput" placeholder="Ask anything..." />
      </label>
      <button class="kb-send" id="kbSend">Send</button>
    </div>
  `;
  wrap.appendChild(modal);
  if(!showLogo){ const logo = modal.querySelector('.kb-logo'); if(logo) logo.style.display = 'none'; }
  
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

  const clickHandler = (e)=>{
    const isOpen = modal.classList.contains('kb-open');
    if (!isOpen) return;
    const path = e.composedPath ? e.composedPath() : [];
    const clickedInside = path.includes(modal) || path.includes(btn) || path.includes(host);
    if (!clickedInside) closePanel();
  };
  document.addEventListener('click', clickHandler);
  if (shadowRoot) shadowRoot.addEventListener('click', clickHandler);
  document.addEventListener('keydown', (e)=>{ if (e.key === 'Escape') closePanel(); });

  function addMsg(role, html){
    const node = document.createElement('div');
    node.className = 'kb-msg ' + role;
    node.innerHTML = `
      <div><div class="kb-bubble">${html}</div><div class="kb-meta">${role==='user'?'You':'Assistant'}</div></div>`;
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
          // Hard filter control frames before any parsing
          if (/"type"\s*:\s*"(start|end)"/.test(dataContent)) { continue; }
                if (dataContent === '[DONE]') continue;
          if (dataContent.startsWith('[ERROR]')) throw new Error(dataContent.replace('[ERROR] ', ''));
          try {
            const json = JSON.parse(dataContent);
            if (json.type === 'content') {
              acc += (json.content || '');
              } else {
              // ignore other frame types
            }
          }
          catch { acc += dataContent; }
          const clean = acc.replace(/\[source: .+?\]/g, '').replace(/^Getting your response\.\.\.?\s*/, '');
          aiBubble.innerHTML = `<div style="white-space:pre-wrap;word-break:break-word;overflow-wrap:anywhere">${clean}</div>`;
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
