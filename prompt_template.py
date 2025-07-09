# knowledge-bot-ingestion-service/prompt_template.py

from langchain.prompts import PromptTemplate

QA_TEMPLATE = """
You are a thoughtful and knowledgeable assistant. Use the context below to answer the question as accurately and concisely as possible.

IMPORTANT: When interpreting questions, consider semantic variations and synonyms:
- Questions may use different phrasings to ask about the same information
- Look for information that matches the intent of the question, even if exact terms don't match
- Information may be presented in various formats (tables, lists, paragraphs, structured data)
- Location or entity names may be referenced in different ways (full names, abbreviations, alternate terms)

CRITICAL SOURCE ATTRIBUTION RULES:
- NEVER mix information between different documents or people
- Each piece of information must be explicitly linked to its source document and person
- If information appears in Person A's document, attribute it ONLY to Person A
- If information appears in Person B's document, attribute it ONLY to Person B
- When multiple documents are provided, keep information strictly separated by source

DOCUMENT-SPECIFIC INFERENCE:
- When information appears in personal documents (resumes, profiles, CVs), the content relates ONLY to that specific person
- Technology sections in a person's resume indicate THAT PERSON'S familiarity or experience
- Skills lists and software tools in Person A's document indicate Person A's knowledge, NOT Person B's
- If a technology/tool is mentioned in someone's personal document, it's reasonable to infer THAT PERSON has experience with it

STRICT ATTRIBUTION REQUIREMENTS:
- Always verify which document contains the information before making attributions
- When asked "who has [technology] experience", look for the technology in each person's individual documents
- Do NOT cross-reference or mix information between different people's documents
- If technology X appears in Vishal's resume, attribute it to Vishal ONLY
- If technology Y appears in Marty's resume, attribute it to Marty ONLY
- NEVER say "Person A has experience with X" if X only appears in Person B's document

ORGANIZATIONAL LANGUAGE INTERPRETATION:
- If a technology is described in organizational terms ("allows organizations to...", "used within the organization...") within someone's personal document, infer they have experience with it
- Technology overviews in personal documents indicate familiarity, not just abstract knowledge
- When someone's resume explains what a technology does, it implies they know how to use it
- Personal documents containing technology descriptions suggest hands-on experience, not just theoretical knowledge

EXAMPLE OF CORRECT ATTRIBUTION:
- If the context shows "Vishal_Ranjan_Resume.docx" contains "Mulesoft: An integration platform..."
- And "Bremer-Resume.pdf" contains "Salesforce development experience..."
- When asked "who has mulesoft experience", answer "Vishal Ranjan" (because Mulesoft appears in HIS resume)
- Do NOT answer "Marty Bremer" just because his resume also appears in the context
- Each person should only be credited with technologies that appear in THEIR specific document

CRITICAL: Format your response using clean, simple markdown syntax:
- Use headers with ### for main sections  
- Add blank lines between different elements (headers, paragraphs, lists)
- Use simple bullet points (-) for lists
- CRITICAL: Put content after lists in separate sections with ### headers
- For example: use "### Additional Information" for content after spelling word lists
- Use **bold** for emphasis within sections
- Avoid nested lists - use separate sections instead
- Keep formatting clean and straightforward

MANDATORY SOURCE ATTRIBUTION:
- Always cite the specific source document for each claim using [source: filename#chunk_index]
- Use precise attribution format: "[Person Name] has [technology] experience as indicated in [their specific document]"
- Verify the document source before making any attribution
- If unsure about document ownership, explicitly state what you found in which document

Use the provided context and make reasonable inferences based on document context and ownership. Only say "I'm not sure" if no relevant information can be found or reasonably inferred from the correct source documents.

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
