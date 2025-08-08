# Knowledge Bot Widget Integration Guide

## Overview
Embed a glassmorphism chat widget on any website. It preserves token handling, session management, markdown rendering, DOM sanitization, and streaming.

## Prerequisites
- Token from CMS → Bot Management → copy the bot’s widget token
- Optional: `DOMPurify` for sanitization
- Recommended meta for mobile: `<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">`

## Production embed
```html
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.0.2/dist/purify.min.js" defer></script>
<script
  src="https://knowledge-bot-retrieval.onrender.com/static/widget.js"
  data-token="YOUR_WIDGET_TOKEN_HERE"
  data-bubble-title="Ask ClearlyClear"
  data-search-title="ClearlyClear Concierge"
  data-show-logo="false"
  defer
></script>
```

## Local development
Run your backend on port 8000, then:
```html
<script
  src="http://localhost:8000/static/widget.js"
  data-token="YOUR_LOCAL_TEST_TOKEN"
  data-bubble-title="Chat"
  data-search-title="Assistant"
  data-show-logo="false"
  defer
></script>
```
Notes:
- The widget auto-detects `localhost`/`127.0.0.1` for API calls; otherwise uses production.
- If you use a different local port, adjust the script URL.

## Attributes
- `data-token` (required): your bot’s widget token
- `data-bubble-title` (optional): text in floating bubble; if omitted, an icon is shown
- `data-search-title` (optional): title in the chat header; blank if omitted
- `data-show-logo` (optional): `"true"` to show logo; hidden by default

## Behavior
- Opens from a bottom-right bubble; panel grows upward only and is capped to screen height on desktop and mobile (uses `vh/dvh`)
- Enter to send; or click Send
- Streams responses via SSE/JSON; control frames are filtered out
- Markdown rendered; sources appear in a collapsible “Show Sources” section
- If `DOMPurify` is present, all rendered content is sanitized

## Styling and isolation
- Widget ships with scoped `.kb-*` styles
- If your site applies heavy resets, ensure it doesn’t override `.kb-*` rules (especially buttons/inputs inside the widget)

## Mobile specifics
- Uses `viewport-fit=cover` and `dvh` when supported
- Input font-size is 16px on small screens to avoid iOS zoom on focus
- Header (with close button) stays accessible; content area scrolls

## Token source
- In CMS → Bot Management → select bot → copy token → set `data-token`
- If you rotate tokens, update the attribute only—no other changes needed

## Troubleshooting
- Blank/white text: ensure site CSS doesn’t force color overrides on `.kb-*`
- CORS: allow your site’s origin on the backend if embedding cross-origin
- Connection issues locally: confirm server at `http://localhost:8000` is running

## Security
- Multi-tenant isolation enforced server-side
- Client sanitization available via `DOMPurify` 