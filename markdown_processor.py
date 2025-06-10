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
    
    Simple, effective approach: filter out markdown symbols immediately
    and send clean tokens to users in real-time.
    
    Args:
        token: Current streaming token from LLM
        buffer: Accumulated buffer from previous tokens
        
    Returns:
        Tuple of (processed_token_to_send, updated_buffer)
    """
    # Simple real-time filtering - remove obvious markdown symbols
    processed_token = token
    
    # Filter out markdown header symbols
    if token.strip() in ['#', '##', '###']:
        return "", buffer  # Don't send header symbols
    
    # Filter out asterisks used for bold/italic
    if token.strip() in ['*', '**']:
        return "", buffer  # Don't send bold/italic symbols
    
    # Filter out tokens that are just asterisks with spaces
    if token.replace(' ', '').replace('\n', '') in ['*', '**']:
        return "", buffer
    
    # Filter out backticks
    if token == '`':
        return "", buffer
    
    # Send normal content tokens immediately
    return processed_token, buffer 