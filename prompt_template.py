# knowledge-bot-ingestion-service/prompt_template.py

from langchain.prompts import PromptTemplate

QA_TEMPLATE = """
You are a thoughtful and knowledgeable assistant. Use the context below to answer the question as accurately and concisely as possible.

When formatting your response:
- Convert any markdown headers (### Section Name) into clean section titles followed by a colon
- Convert markdown bold (**text**) into plain text without asterisks
- Use simple bullet points (-) and proper indentation for lists
- Create clear, readable sections without markdown formatting symbols
- Make the response well-organized and easy to scan

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
