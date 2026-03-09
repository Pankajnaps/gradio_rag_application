"""
Chatbot with RAG Router using OpenAI ChatCompletions API
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

from rag_pipeline import run_ingestion, retrieve_documents

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------------------------------------
# OpenAI Chat Function
# -------------------------------------------------

def chat_with_openai(prompt):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


# -------------------------------------------------
# Intent Detection
# -------------------------------------------------

def detect_intent(query):

    prompt = f"""
Classify the user query into one category.

Categories:
1. IPL_RULE_QUERY → Questions about IPL rules or regulations
2. GENERAL_QUERY → Any other question

Return ONLY the category name.

Query: {query}
"""

    result = chat_with_openai(prompt)

    return result.strip()


# -------------------------------------------------
# RAG Answer
# -------------------------------------------------

def answer_from_rag(query, vectordb):

    docs = retrieve_documents(query, vectordb)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an expert assistant answering IPL rulebook questions.

Use ONLY the provided context.

Context:
{context}

Question:
{query}

If answer not found in context say:
"I cannot find this rule in the IPL rulebook."
"""

    return chat_with_openai(prompt)


# -------------------------------------------------
# General Chat
# -------------------------------------------------

def general_chat(query):

    prompt = f"""
You are a helpful AI assistant.

User question:
{query}
"""

    return chat_with_openai(prompt)


# -------------------------------------------------
# Chatbot Router
# -------------------------------------------------

def chatbot(query, vectordb):

    intent = detect_intent(query)

    print("\nDetected Intent:", intent)

    if "IPL_RULE_QUERY" in intent:
        return answer_from_rag(query, vectordb)

    return general_chat(query)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("\nLoading RAG system...\n")

    vectordb = run_ingestion()

    print("\nChatbot Ready\n")

    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        answer = chatbot(query, vectordb)

        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    main()