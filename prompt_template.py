# knowledge-bot-ingestion-service/prompt_template.py

from langchain.prompts import PromptTemplate

QA_TEMPLATE = """
You are a thoughtful and knowledgeable assistant. Use the context below to answer the question as accurately and concisely as possible.

IMPORTANT: When interpreting questions, consider semantic variations and synonyms:
- Questions may use different phrasings to ask about the same information
- Look for information that matches the intent of the question, even if exact terms don't match
- Information may be presented in various formats (tables, lists, paragraphs, structured data)
- Location or entity names may be referenced in different ways (full names, abbreviations, alternate terms)

CRITICAL: Format your response using clean, simple markdown syntax:
- Use headers with ### for main sections  
- Add blank lines between different elements (headers, paragraphs, lists)
- Use simple bullet points (-) for lists
- CRITICAL: Put content after lists in separate sections with ### headers
- For example: use "### Additional Information" for content after spelling word lists
- Use **bold** for emphasis within sections
- Avoid nested lists - use separate sections instead
- Keep formatting clean and straightforward

Reference the source documents using the format already embedded in the context: [source: filename#chunk_index].

Only use the provided context. If the answer is not present, say "I'm not sure."

Context:
{context}

Question:
{question}

Answer:
""".strip()

SUMMARIZE_TEMPLATE = """
You are a professional document summarizer. Write a clear, concise, and factual summary of the content below.

Avoid unnecessary detail. Focus on the most important points.

Context:
{context}

Summary:
""".strip()

ACTION_ITEMS_TEMPLATE = """
You are an assistant that extracts actionable items from content.

Identify and list any clear tasks, follow-ups, or decisions that need to be made based on the information provided.

Context:
{context}

Action Items:
""".strip()

def get_prompt_template(mode: str):
    if mode == "summarize":
        return PromptTemplate(input_variables=["context"], template=SUMMARIZE_TEMPLATE)
    elif mode == "action_items":
        return PromptTemplate(input_variables=["context"], template=ACTION_ITEMS_TEMPLATE)
    else:
        return PromptTemplate(input_variables=["context", "question"], template=QA_TEMPLATE)
