(() => {
  const token = document.currentScript.getAttribute("data-token");
  const API_URL = "https://knowledge-bot-retrieval.onrender.com";

  const STYLE = `
    .kb-btn{position:fixed;bottom:24px;right:24px;width:56px;height:56px;background:#2563eb;border-radius:9999px;color:#fff;font-size:24px;display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:9999}
    .kb-modal{position:fixed;bottom:90px;right:24px;width:360px;max-height:80vh;background:#fff;border-radius:16px;padding:16px;box-shadow:0 4px 20px rgba(0,0,0,.1);font-family:sans-serif;z-index:9999;display:none;flex-direction:column}
    .kb-modal textarea{resize:none;padding:10px;border:1px solid #ccc;border-radius:8px;width:100%}
    .kb-modal button{margin-top:8px;background:#2563eb;color:#fff;padding:10px;border:none;border-radius:8px;cursor:pointer}
    .kb-answer{margin-top:12px;background:#f9fafb;padding:10px;border-radius:8px;font-size:14px;white-space:pre-wrap}
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
    <textarea rows="3" placeholder="Ask something..."></textarea>
    <button>Ask</button>
    <div class="kb-answer"></div>
  `;
  document.body.appendChild(modal);

  btn.onclick = () => {
    modal.style.display = modal.style.display === "flex" ? "none" : "flex";
  };

  modal.querySelector("button").onclick = async () => {
    const question = modal.querySelector("textarea").value.trim();
    const output = modal.querySelector(".kb-answer");
    if (!question) return;

    output.textContent = "Thinking...";

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ question })
      });
      const data = await res.json();
      output.textContent = data.answer || data.error || "No response.";
    } catch (err) {
      output.textContent = `Error: ${err.message}`;
    }
  };
})();
