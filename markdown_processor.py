import re
import logging

logger = logging.getLogger(__name__)

def process_markdown_to_clean_text(markdown_text: str) -> str:
    """
    Convert markdown formatting to clean, readable text.
    
    This is a production-grade post-processor that handles:
    - Headers (###, ##, #) → Clean section titles
    - Bold (**text**) → Plain text
    - Lists and bullet points
    - Proper spacing and readability
    
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
    Process streaming tokens for real-time markdown conversion.
    
    This handles the challenge of converting markdown in real-time streaming
    by maintaining a buffer to detect markdown patterns that span multiple tokens.
    
    Args:
        token: Current streaming token from LLM
        buffer: Accumulated buffer from previous tokens
        
    Returns:
        Tuple of (processed_token_to_send, updated_buffer)
    """
    # Add current token to buffer
    buffer += token
    
    # For real-time streaming, we can only safely process completed patterns
    # We'll send most tokens immediately and only buffer when we detect potential markdown
    
    # Check if we're in the middle of a potential markdown pattern
    markdown_patterns = [
        r'\*\*[^*]*$',  # Incomplete bold
        r'\*[^*]*$',    # Incomplete italic  
        r'^#{1,3}\s[^:\n]*$',  # Incomplete header
        r'`[^`]*$',     # Incomplete code
    ]
    
    # If buffer matches incomplete pattern, hold the token
    for pattern in markdown_patterns:
        if re.search(pattern, buffer):
            return "", buffer
    
    # Check for completed patterns and process them
    if re.search(r'\*\*.*?\*\*', buffer):
        # Process completed bold
        processed_buffer = re.sub(r'\*\*(.*?)\*\*', r'\1', buffer)
        return processed_buffer, ""
    
    if re.search(r'^#{1,3}\s.*?:', buffer, re.MULTILINE):
        # Process completed header
        processed_buffer = re.sub(r'^(#{1,3})\s(.+)$', r'\2:', buffer, flags=re.MULTILINE)
        return processed_buffer, ""
    
    # If no markdown patterns detected, send the token
    return buffer, "" 