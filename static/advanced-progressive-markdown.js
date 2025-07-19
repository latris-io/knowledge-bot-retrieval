/**
 * Advanced Progressive Markdown Parser
 * True incremental parsing with chunk-aware rendering
 */

class AdvancedProgressiveMarkdown {
  constructor() {
    this.md = markdownit({
      html: true,
      linkify: true,
      typographer: true,
      breaks: true
    });
    
    // State management
    this.contentBuffer = '';
    this.parsedHtml = '';
    this.currentBlockType = null;
    this.blockBuffer = '';
    
    // Performance optimization
    this.parseCache = new Map();
    this.lastParsedLength = 0;
  }

  /**
   * Process chunk with intelligent incremental parsing
   */
  processChunk(chunkData) {
    const { type, content, content_type } = chunkData;
    
    if (type === 'start') {
      this.reset();
      return { html: '', action: 'start' };
    }
    
    if (type === 'end') {
      return this.finalizeStream();
    }
    
    if (type === 'error') {
      return { html: `<div class="error">${chunkData.error}</div>`, action: 'error' };
    }
    
    // Handle content chunks
    if (type === 'content' && content) {
      return this.handleProgressiveContent(content, content_type);
    }
    
    return { html: '', action: 'ignore' };
  }

  /**
   * Handle content with smart incremental parsing
   */
  handleProgressiveContent(content, contentType) {
    this.contentBuffer += content;
    
    // Different strategies based on content type
    switch (contentType) {
      case 'header':
        return this.processCompleteBlock(content, 'header');
        
      case 'list_item':
        return this.processListItem(content);
        
      case 'source':
        return this.processSource(content);
        
      case 'text':
      default:
        return this.processTextContent(content);
    }
  }

  /**
   * Process complete markdown blocks that can be parsed immediately
   */
  processCompleteBlock(content, type) {
    const cacheKey = `${type}:${content}`;
    
    if (this.parseCache.has(cacheKey)) {
      const cachedHtml = this.parseCache.get(cacheKey);
      return { html: cachedHtml, action: 'append' };
    }
    
    // Parse the complete block
    const html = this.md.render(content.trim());
    this.parseCache.set(cacheKey, html);
    
    return { html, action: 'append' };
  }

  /**
   * Handle list items with context awareness
   */
  processListItem(content) {
    // Check if this completes a list item
    if (content.includes('\n') || content.endsWith('  ')) {
      // Complete list item - can parse
      const html = this.md.render(content.trim());
      return { html, action: 'append' };
    }
    
    // Incomplete list item - show as raw text for now
    return { 
      html: `<span class="streaming-list">${this.escapeHtml(content)}</span>`, 
      action: 'append',
      needsReparse: true
    };
  }

  /**
   * Process regular text with sentence-boundary awareness
   */
  processTextContent(content) {
    // Strategy: Parse when we have complete sentences or paragraphs
    this.blockBuffer += content;
    
    // Check for complete sentences
    if (this.hasCompleteSentence(this.blockBuffer)) {
      const sentences = this.extractCompleteSentences();
      const html = this.md.render(sentences);
      
      // Update state
      this.blockBuffer = this.blockBuffer.substring(sentences.length);
      
      return { html, action: 'append' };
    }
    
    // Check for paragraph breaks
    if (this.blockBuffer.includes('\n\n')) {
      const paragraphs = this.extractCompleteParagraphs();
      const html = this.md.render(paragraphs);
      
      return { html, action: 'append' };
    }
    
    // Not ready to parse - show as streaming text
    return { 
      html: `<span class="streaming-text">${this.escapeHtml(content)}</span>`, 
      action: 'append',
      needsReparse: true
    };
  }

  /**
   * Check if buffer contains complete sentences
   */
  hasCompleteSentence(text) {
    // Look for sentence endings followed by space or newline
    return /[.!?]\s/.test(text) || /[.!?]$/.test(text.trim());
  }

  /**
   * Extract complete sentences from buffer
   */
  extractCompleteSentences() {
    const match = this.blockBuffer.match(/(.*?[.!?]\s*)/s);
    return match ? match[1] : '';
  }

  /**
   * Extract complete paragraphs from buffer
   */
  extractCompleteParagraphs() {
    const parts = this.blockBuffer.split('\n\n');
    if (parts.length > 1) {
      const completeParagraphs = parts.slice(0, -1).join('\n\n') + '\n\n';
      this.blockBuffer = parts[parts.length - 1];
      return completeParagraphs;
    }
    return '';
  }

  /**
   * Process source citations
   */
  processSource(content) {
    const match = content.match(/\[source: (.+?)\]/);
    if (match) {
      return {
        html: '',
        action: 'source',
        source: match[1]
      };
    }
    return { html: '', action: 'ignore' };
  }

  /**
   * Finalize the stream with complete parsing
   */
  finalizeStream() {
    // Parse any remaining content
    let finalHtml = this.parsedHtml;
    
    if (this.blockBuffer.trim()) {
      const remainingHtml = this.md.render(this.blockBuffer.trim());
      finalHtml += remainingHtml;
    }
    
    // Clean up and extract sources
    const cleanContent = this.contentBuffer.replace(/\[source: .+?\]/g, '');
    const completeHtml = this.md.render(cleanContent);
    
    return {
      html: completeHtml,
      action: 'replace',
      sources: this.extractSources()
    };
  }

  /**
   * Extract all sources from content buffer
   */
  extractSources() {
    const matches = [...this.contentBuffer.matchAll(/\[source: (.+?)\]/g)];
    return [...new Set(matches.map(match => match[1]))];
  }

  /**
   * Reset parser state
   */
  reset() {
    this.contentBuffer = '';
    this.parsedHtml = '';
    this.currentBlockType = null;
    this.blockBuffer = '';
    this.lastParsedLength = 0;
  }

  /**
   * Escape HTML for safe display
   */
  escapeHtml(text) {
    return text.replace(/[&<>"']/g, char => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[char]));
  }
}

/**
 * Enhanced widget integration with progressive parsing
 */
class EnhancedStreamingWidget {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.parser = new AdvancedProgressiveMarkdown();
    this.sources = new Set();
    
    // DOM elements
    this.contentArea = null;
    this.sourcesArea = null;
    this.setupDOM();
  }

  setupDOM() {
    this.container.innerHTML = `
      <div class="kb-content-area"></div>
      <div class="kb-sources-area" style="display: none;">
        <details class="kb-sources">
          <summary>Sources (<span class="source-count">0</span>)</summary>
          <ul class="source-list"></ul>
        </details>
      </div>
    `;
    
    this.contentArea = this.container.querySelector('.kb-content-area');
    this.sourcesArea = this.container.querySelector('.kb-sources-area');
  }

  async processStream(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (!data.trim()) continue;
            
            await this.processChunk(data);
          }
        }
      }
    } catch (error) {
      this.showError(error.message);
    }
  }

  async processChunk(data) {
    try {
      const chunkData = JSON.parse(data);
      const result = this.parser.processChunk(chunkData);
      
      switch (result.action) {
        case 'start':
          this.contentArea.innerHTML = '<div class="kb-spinner">Thinking...</div>';
          this.sources.clear();
          this.updateSourcesDisplay();
          break;
          
        case 'append':
          if (result.html) {
            // Remove spinner if present
            const spinner = this.contentArea.querySelector('.kb-spinner');
            if (spinner) spinner.remove();
            
            // Add new content
            this.contentArea.insertAdjacentHTML('beforeend', result.html);
            this.scrollToBottom();
          }
          break;
          
        case 'replace':
          // Final parsing complete
          this.contentArea.innerHTML = result.html;
          if (result.sources) {
            result.sources.forEach(source => this.sources.add(source));
            this.updateSourcesDisplay();
          }
          this.scrollToBottom();
          break;
          
        case 'source':
          this.sources.add(result.source);
          this.updateSourcesDisplay();
          break;
          
        case 'error':
          this.showError(result.html);
          break;
      }
      
    } catch (parseError) {
      console.warn('Failed to parse chunk:', parseError);
      // Fallback: treat as raw text
      this.contentArea.insertAdjacentHTML('beforeend', this.escapeHtml(data));
    }
  }

  updateSourcesDisplay() {
    if (this.sources.size > 0) {
      this.sourcesArea.style.display = 'block';
      this.container.querySelector('.source-count').textContent = this.sources.size;
      this.container.querySelector('.source-list').innerHTML = 
        Array.from(this.sources).map(source => `<li>${source}</li>`).join('');
    } else {
      this.sourcesArea.style.display = 'none';
    }
  }

  showError(message) {
    this.contentArea.innerHTML = `<div class="kb-error">Error: ${message}</div>`;
  }

  scrollToBottom() {
    this.container.scrollTop = this.container.scrollHeight;
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Usage
const streamingWidget = new EnhancedStreamingWidget('answerBox');

// In your fetch handler:
async function handleStreamingResponse(response) {
  await streamingWidget.processStream(response);
} 