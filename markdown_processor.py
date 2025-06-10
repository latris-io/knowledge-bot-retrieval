import re
import logging

logger = logging.getLogger(__name__)

def process_markdown_to_html(markdown_text: str) -> str:
    """
    Convert markdown formatting to HTML for rich user experience.
    
    This is a production-grade post-processor that converts:
    - Headers (###, ##, #) → HTML headers
    - Bold (**text**) → <strong> tags
    - Italic (*text*) → <em> tags
    - Lists and bullet points → <ul>/<li> tags
    - Proper HTML structure
    
    Args:
        markdown_text: Raw markdown text from LLM
        
    Returns:
        HTML formatted text for rich user experience
    """
    if not markdown_text:
        return ""
    
    html_text = markdown_text.strip()
    
    # Convert headers to HTML
    html_text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_text, flags=re.MULTILINE)
    html_text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_text, flags=re.MULTILINE)
    html_text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_text, flags=re.MULTILINE)
    
    # Convert bold text to HTML
    html_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_text)
    
    # Convert italic text to HTML
    html_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html_text)
    
    # Convert code inline
    html_text = re.sub(r'`([^`]+)`', r'<code>\1</code>', html_text)
    
    # Convert bullet points to HTML lists
    # First, identify bullet point sections
    lines = html_text.split('\n')
    processed_lines = []
    in_list = False
    
    for line in lines:
        # Check if this is a bullet point
        if re.match(r'^[\s]*[-*+]\s+', line):
            if not in_list:
                processed_lines.append('<ul>')
                in_list = True
            # Extract the bullet content
            content = re.sub(r'^[\s]*[-*+]\s+', '', line)
            processed_lines.append(f'<li>{content}</li>')
        else:
            if in_list:
                processed_lines.append('</ul>')
                in_list = False
            if line.strip():  # Only add non-empty lines
                processed_lines.append(line)
    
    # Close any open list
    if in_list:
        processed_lines.append('</ul>')
    
    html_text = '\n'.join(processed_lines)
    
    # Convert paragraphs (double newlines become <br><br>)
    html_text = re.sub(r'\n\n+', '<br><br>', html_text)
    
    # Convert single newlines to <br> for line breaks
    html_text = re.sub(r'\n', '<br>', html_text)
    
    # Convert links
    html_text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html_text)
    
    # Clean up excessive breaks
    html_text = re.sub(r'(<br>){3,}', '<br><br>', html_text)
    
    return html_text.strip()


def process_markdown_to_clean_text(markdown_text: str) -> str:
    """
    Convert markdown formatting to clean, readable text for conversation history.
    
    This strips markdown symbols to create clean text for follow-up context.
    
    Args:
        markdown_text: Raw markdown text from LLM
        
    Returns:
        Clean, formatted text without markdown symbols
    """
    if not markdown_text:
        return ""
    
    processed_text = markdown_text.strip()
    
    # Convert headers to clean section titles
    # ### Section Name → Section Name:
    processed_text = re.sub(r'^### (.+)$', r'\1:', processed_text, flags=re.MULTILINE)
    processed_text = re.sub(r'^## (.+)$', r'\1:', processed_text, flags=re.MULTILINE)
    processed_text = re.sub(r'^# (.+)$', r'\1:', processed_text, flags=re.MULTILINE)
    
    # Convert bold text to plain text
    # **bold text** → bold text
    processed_text = re.sub(r'\*\*(.*?)\*\*', r'\1', processed_text)
    
    # Convert italic text to plain text  
    # *italic* → italic
    processed_text = re.sub(r'\*(.*?)\*', r'\1', processed_text)
    
    # Clean up bullet points - keep simple dashes
    # Ensure consistent spacing for bullets
    processed_text = re.sub(r'^[\s]*[-*+]\s+', '- ', processed_text, flags=re.MULTILINE)
    
    # Clean up numbered lists
    processed_text = re.sub(r'^\s*\d+\.\s+', '- ', processed_text, flags=re.MULTILINE)
    
    # Remove excessive whitespace but preserve paragraph breaks
    # Replace multiple consecutive whitespace with single space
    processed_text = re.sub(r'[ \t]+', ' ', processed_text)
    
    # Preserve paragraph breaks (double newlines)
    processed_text = re.sub(r'\n{3,}', '\n\n', processed_text)
    
    # Clean up spacing around colons (section headers)
    processed_text = re.sub(r'\s*:\s*\n', ':\n', processed_text)
    
    # Ensure proper spacing after section headers
    processed_text = re.sub(r':(\n)([^\n-])', r':\n\n\2', processed_text)
    
    # Clean up any remaining markdown artifacts
    processed_text = re.sub(r'`([^`]+)`', r'\1', processed_text)  # Remove code backticks
    processed_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', processed_text)  # Convert links to text
    
    # Final cleanup
    processed_text = processed_text.strip()
    
    return processed_text


def process_streaming_token(token: str, buffer: str = "") -> tuple[str, str]:
    """
    Process streaming tokens for real-time HTML conversion.
    
    Converts markdown to HTML in real-time for rich formatting.
    
    Args:
        token: Current streaming token from LLM
        buffer: Accumulated buffer from previous tokens
        
    Returns:
        Tuple of (processed_token_to_send, updated_buffer)
    """
    # Add current token to buffer
    buffer += token
    
    # Check for completed markdown patterns and convert to HTML
    
    # Bold text: **text** → <strong>text</strong>
    if '**' in buffer:
        # Count asterisks to find complete bold patterns
        asterisk_count = buffer.count('**')
        if asterisk_count >= 2:  # We have at least one complete bold pattern
            # Process the first complete bold pattern
            match = re.search(r'\*\*(.*?)\*\*', buffer)
            if match:
                before = buffer[:match.start()]
                bold_content = match.group(1)
                after = buffer[match.end():]
                
                # Send the before content + HTML bold
                output = before + f"<strong>{bold_content}</strong>"
                return output, after
    
    # Headers: ### text → <h3>text</h3>
    if buffer.strip().startswith(('#', '##', '###')) and '\n' in buffer:
        lines = buffer.split('\n', 1)
        if len(lines) >= 2:
            header_line = lines[0]
            rest = lines[1]
            
            if header_line.startswith('### '):
                header_content = header_line[4:]
                html_header = f"<h3>{header_content}</h3>"
                return html_header + "\n", rest
            elif header_line.startswith('## '):
                header_content = header_line[3:]
                html_header = f"<h2>{header_content}</h2>"
                return html_header + "\n", rest
            elif header_line.startswith('# '):
                header_content = header_line[2:]
                html_header = f"<h1>{header_content}</h1>"
                return html_header + "\n", rest
    
    # If we have a complete line with bullet points
    if buffer.strip().startswith(('-', '*', '+')) and '\n' in buffer:
        lines = buffer.split('\n', 1)
        if len(lines) >= 2:
            bullet_line = lines[0]
            rest = lines[1]
            
            # Convert bullet to HTML
            content = re.sub(r'^[\s]*[-*+]\s+', '', bullet_line)
            html_bullet = f"<li>{content}</li>"
            return html_bullet + "\n", rest
    
    # If buffer is getting long without patterns, send as-is
    if len(buffer) > 100:
        return buffer, ""
    
    # Otherwise, keep buffering
    return "", buffer 