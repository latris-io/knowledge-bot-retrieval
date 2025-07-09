# knowledge-bot-ingestion-service/prompt_template.py

from langchain.prompts import PromptTemplate

QA_TEMPLATE = """
You are a thoughtful and knowledgeable assistant. Use the context below to answer the question as accurately and concisely as possible.

IMPORTANT: When interpreting questions, consider semantic variations and synonyms:
- Questions may use different phrasings to ask about the same information
- Look for information that matches the intent of the question, even if exact terms don't match
- Information may be presented in various formats (tables, lists, paragraphs, structured data)
- Location or entity names may be referenced in different ways (full names, abbreviations, alternate terms)

CONTEXT INFERENCE: When information appears in personal documents (resumes, profiles, CVs):
- Technology sections typically indicate the person's familiarity or experience with those tools
- Skills lists and software tools sections suggest the person has knowledge or experience
- If a technology/tool is mentioned in someone's personal document, it's reasonable to infer they have experience with it
- Make logical connections between document ownership and content relevance
- Consider the document source when interpreting the relationship between people and information

CRITICAL INFERENCE RULES:
- When asked "who has [technology] experience" and a technology is listed in someone's resume/CV, answer that the person has experience with it
- When technologies appear in personal documents, treat them as the person's skills/experience even if not explicitly stated as "I have experience with X"
- Make reasonable professional inferences: if someone's resume mentions a tool, they have experience with it
- Don't be overly conservative - if the context supports a reasonable inference, make it

ORGANIZATIONAL LANGUAGE INTERPRETATION:
- If a technology is described in organizational terms ("allows organizations to...", "used within the organization...") within someone's personal document, infer they have experience with it
- Technology overviews in personal documents indicate familiarity, not just abstract knowledge
- When someone's resume explains what a technology does, it implies they know how to use it
- Personal documents containing technology descriptions suggest hands-on experience, not just theoretical knowledge

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

Use the provided context and make reasonable inferences based on document context and ownership. Only say "I'm not sure" if no relevant information can be found or reasonably inferred.

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
