# Knowledge Bot Widget Integration Guide

## Overview
Embed a glassmorphism chat widget on any website. It preserves token handling, session management, markdown rendering, DOM sanitization, and streaming.

## Prerequisites
- Token from CMS → Bot Management → copy the bot’s widget token
- Optional: `DOMPurify` for sanitization
- Recommended meta for mobile: `<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">`

## Core snippet (production)
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

## Attributes
- `data-token` (required): your bot’s widget token
- `data-bubble-title` (optional): text in floating bubble; if omitted, an icon is shown
- `data-search-title` (optional): title in the chat header; blank if omitted
- `data-show-logo` (optional): `"true"` to show logo; hidden by default

## Behavior
- Opens from a bottom-right bubble; panel grows upward only and is capped to screen height on desktop and mobile (`vh/dvh`)
- Enter to send; or click Send
- Streams responses via SSE/JSON; control frames are filtered out
- Markdown rendered; sources appear in a collapsible “Show Sources” section
- If `DOMPurify` is present, all rendered content is sanitized

---

## Platform-specific instructions

### Plain HTML sites
- Paste the Core snippet before `</body>` (recommended) or inside `<head>` with `defer`.
- Ensure no global CSS overrides `.kb-*` classes (buttons/inputs).

### WordPress
- Appearance → Theme File Editor → `footer.php` → paste Core snippet before `</body>` and Update file.
- Alternatively: Plugins → “Code Snippets” → Add New → Code Type: HTML → Location: Front-end Footer → paste snippet → Save & Activate.
- Block editor: Add a “Custom HTML” block site-wide via Widgets (Footer) and paste the snippet.

### Webflow
- Project Settings → Custom Code → Footer Code → paste Core snippet → Save → Publish.
- Or per page: Add an Embed element at the end of the page and paste the snippet.

### Squarespace
- Settings → Advanced → Code Injection → Footer → paste Core snippet → Save.
- Or page-level “Code” block at the bottom of the page.

### Wix
- Settings → Advanced → Custom Code → Add Custom Code → paste Core snippet.
  - Place Code: All pages
  - Location: Body – end
  - Load: On All Devices
- Publish.

### Shopify
- Online Store → Themes → Edit code → `layout/theme.liquid`.
- Paste Core snippet just before `</body>` and Save.
- If using Dawn/Online Store 2.0, you can also add via “theme.liquid” or a section that renders site-wide.

### React / Next.js
- Next.js 13+ (App Router), in `app/layout.tsx`:
```tsx
import Script from 'next/script';
export default function RootLayout({ children }){
  return (
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
        <Script src="https://cdn.jsdelivr.net/npm/dompurify@3.0.2/dist/purify.min.js" strategy="afterInteractive" />
        <Script
          src="https://knowledge-bot-retrieval.onrender.com/static/widget.js"
          data-token="YOUR_WIDGET_TOKEN_HERE"
          data-bubble-title="Ask ClearlyClear"
          data-search-title="ClearlyClear Concierge"
          data-show-logo="false"
          strategy="afterInteractive"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
```
- Next.js (Pages Router), in `pages/_document.tsx` or `_app.tsx` using `next/script` with `afterInteractive`.

### Vue / Nuxt
- Nuxt 3 (`nuxt.config.ts`):
```ts
export default defineNuxtConfig({
  app: {
    head: {
      meta: [{ name: 'viewport', content: 'width=device-width, initial-scale=1, viewport-fit=cover' }],
      script: [
        { src: 'https://cdn.jsdelivr.net/npm/dompurify@3.0.2/dist/purify.min.js', defer: true },
        {
          src: 'https://knowledge-bot-retrieval.onrender.com/static/widget.js',
          defer: true,
          'data-token': 'YOUR_WIDGET_TOKEN_HERE',
          'data-bubble-title': 'Ask ClearlyClear',
          'data-search-title': 'ClearlyClear Concierge',
          'data-show-logo': 'false'
        }
      ]
    }
  }
});
```
- Vue SPA (no Nuxt): add the two `<script>` tags in `public/index.html` before `</body>`.

### Google Tag Manager (GTM)
1) Create a “Custom HTML” tag with the Core snippet (you can omit the `<meta>` tag in GTM). 
2) Trigger: All Pages. 
3) Set “Support document.write” OFF. 
4) Optional: add a second Custom HTML tag for `DOMPurify` (or include it inside the widget tag).

---

## Content Security Policy (CSP)
If you enforce CSP, add allowances similar to:
```http
Content-Security-Policy:
  script-src 'self' https://knowledge-bot-retrieval.onrender.com https://cdn.jsdelivr.net;
  style-src  'self' 'unsafe-inline';
  connect-src 'self' https://knowledge-bot-retrieval.onrender.com http://localhost:8000;
  img-src    'self' data: blob:;
  font-src   'self' data:;
```
Notes:
- The widget injects a `<style>` block; some CSPs require `'unsafe-inline'` in `style-src`.
- If hosting the widget from a different domain, include that origin.

## Styling and isolation
- Widget ships with scoped `.kb-*` styles; avoid global resets that target `button`, `input`, etc., inside `.kb-*`.

## Mobile specifics
- Uses `viewport-fit=cover` and `dvh` when supported for correct height and safe areas.
- Input font-size is 16px on small screens to avoid iOS zoom on focus.
- Header (with close button) remains visible; content scrolls within the panel.

## Token source
- In CMS → Bot Management → select bot → copy token → set `data-token`.
- If you rotate tokens, update the attribute only—no other changes needed.

## Troubleshooting
- Blank/white text: ensure site CSS doesn’t force color overrides on `.kb-*`.
- CORS: allow your site’s origin on the backend if embedding cross-origin.
- Local dev: confirm backend at `http://localhost:8000` is running.
- Reset local session: remove `kb-chat-session` from `localStorage`.

## Accessibility
- Bubble uses `aria-label` and `aria-expanded`. Modal uses `role="dialog"`.
- Outside-click and Esc to close. Focus moves to input on open.

## Uninstall
- Remove the inserted `<script>` tag(s). Clear `localStorage` key `kb-chat-session` if desired. 