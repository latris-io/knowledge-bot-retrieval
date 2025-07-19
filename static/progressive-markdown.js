/**
 * Progressive Markdown Parser for Smart Streaming
 * Integrates with our JSON chunk format for optimal performance
 */

class SmartMarkdownRenderer {
  constructor() {
    this.md = markdownit({
      html: true,
      linkify: true,
      typographer: true,
      breaks: true
    });
    
    this.contentBuffer = '';
    this.renderedChunks = [];
    this.chunkCache = new Map();
  }

  /**
   * Process a streaming chunk based on its content type
   */
  processChunk(chunkData) {
    const { id, type, content, content_type, final } = chunkData;
    
    switch (type) {
      case 'start':
        this.reset();
        return { html: '', action: 'start' };
        
      case 'content':
        return this.handleContent(id, content, content_type);
        
      case 'end':
        return this.finalize();
        
      case 'error':
        return { html: `<div class="error">Error: ${chunkData.error}</div>`, action: 'error' };
    }
  }

  /**
   * Handle content chunks with intelligent parsing strategy
   */
  handleContent(id, content, contentType) {
    this.contentBuffer += content;
    
    // Strategy: Different parsing approaches based on content type
    switch (contentType) {
      case 'header':
        return this.parseImmediately(content, 'header');
        
      case 'list_item':
        return this.parseImmediately(content, 'list_item');
        
      case 'source':
        return this.parseSource(content);
        
      case 'text':
      default:
        return this.accumulateAndParse(content);
    }
  }

  /**
   * Parse complete structural elements immediately
   */
  parseImmediately(content, type) {
    const cacheKey = `${type}:${content}`;
    
    if (this.chunkCache.has(cacheKey)) {
      return { html: this.chunkCache.get(cacheKey), action: 'append' };
    }
    
    let html;
    switch (type) {
      case 'header':
        // Headers are complete and can be parsed immediately
        html = this.md.render(content);
        break;
        
      case 'list_item':
        // List items might need special handling
        html = this.md.render(content);
        break;
    }
    
    this.chunkCache.set(cacheKey, html);
    return { html, action: 'append' };
  }

  /**
   * Handle source citations separately
   */
  parseSource(content) {
    // Extract source information: [source: filename#chunk]
    const sourceMatch = content.match(/\[source: (.+?)\]/);
    if (sourceMatch) {
      const sourceRef = sourceMatch[1];
      return {
        html: '', // Don't render sources inline
        action: 'source',
        source: sourceRef
      };
    }
    return { html: '', action: 'ignore' };
  }

  /**
   * Accumulate text content and parse strategically
   */
  accumulateAndParse(content) {
    // For regular text, we can either:
    // 1. Parse incrementally if content seems complete
    // 2. Accumulate and parse at sentence boundaries
    // 3. Show raw text and parse at the end
    
    // Strategy: Show raw text during streaming, parse at end
    return { 
      html: `<span class="streaming-text">${this.escapeHtml(content)}</span>`, 
      action: 'append' 
    };
  }

  /**
   * Finalize parsing when stream completes
   */
  finalize() {
    // Remove source markers for final parsing
    const cleanContent = this.contentBuffer.replace(/\[source: .+?\]/g, '');
    
    // Parse the complete markdown
    const finalHtml = this.md.render(cleanContent);
    
    return { 
      html: finalHtml, 
      action: 'replace',
      sources: this.extractSources()
    };
  }

  /**
   * Extract all source references
   */
  extractSources() {
    const sourceMatches = [...this.contentBuffer.matchAll(/\[source: (.+?)\]/g)];
    return [...new Set(sourceMatches.map(match => match[1]))];
  }

  /**
   * Reset for new stream
   */
  reset() {
    this.contentBuffer = '';
    this.renderedChunks = [];
    // Keep cache for performance
  }

  /**
   * Escape HTML for safe raw text display
   */
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Usage example
const markdownRenderer = new SmartMarkdownRenderer();

async function handleSmartStream(response) {
  const answerBox = document.getElementById('answerBox');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let sources = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (!data.trim()) continue;
        
        try {
          const chunkData = JSON.parse(data);
          const result = markdownRenderer.processChunk(chunkData);
          
          switch (result.action) {
            case 'start':
              answerBox.innerHTML = '<div class="kb-spinner"></div>';
              break;
              
            case 'append':
              if (result.html) {
                answerBox.insertAdjacentHTML('beforeend', result.html);
              }
              break;
              
            case 'replace':
              // Final parsing complete - replace with properly formatted content
              const sourcesHtml = result.sources.length 
                ? `<details class="kb-sources">
                     <summary>Sources (${result.sources.length})</summary>
                     <ul>${result.sources.map(src => `<li>${src}</li>`).join('')}</ul>
                   </details>`
                : '';
              
              answerBox.innerHTML = result.html + sourcesHtml;
              break;
              
            case 'source':
              sources.push(result.source);
              break;
              
            case 'error':
              answerBox.innerHTML = result.html;
              break;
          }
          
          answerBox.scrollTop = answerBox.scrollHeight;
          
        } catch (parseError) {
          console.warn('Failed to parse chunk:', parseError);
          // Fallback to raw text
        }
      }
    }
  }
} 