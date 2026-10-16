
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()


def extract_answer(response):
    """
    Extract readable text from Gemini's response.
    Handles both string and list-based response formats.
    """
    answer = response.content

    if isinstance(answer, str):
        return answer

    if isinstance(answer, list):
        text_parts = []

        for item in answer:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))

            elif isinstance(item, str):
                text_parts.append(item)

        return "\n".join(text_parts)

    return str(answer)


def main():
    # 1. Verify API key
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Check your .env file."
        )

    # 2. Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2"
    )

    # 3. Load the existing ChromaDB vector database
    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
        collection_name="sample_policies",
    )

    # 4. Create retriever
    retriever = db.as_retriever(
        search_kwargs={"k": 4}
    )

    # 5. Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
    )

    # 6. Create prompt template
    prompt = ChatPromptTemplate.from_template("""
You are a helpful document assistant.

Answer the question using only the context provided below.
If the answer is not present in the context, say:
"I could not find the answer in the provided documents."

Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
""")

    # 7. Start chatbot
    print("=" * 50)
    print("       RAG DOCUMENT CHATBOT")
    print("=" * 50)
    print("Ask questions about your PDF documents.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Chatbot closed. Goodbye!")
                break

            if not question:
                continue

            # 8. Retrieve relevant document chunks
            docs = retriever.invoke(question)

            if not docs:
                print(
                    "\nBot: I could not find relevant information "
                    "in the provided documents."
                )
                continue

            # 9. Build context from retrieved chunks
            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            # 10. Generate answer using Gemini
            chain = prompt | llm

            response = chain.invoke({
                "context": context,
                "question": question,
            })

            # 11. Extract and print only readable text
            answer = extract_answer(response)

            print("\nBot:", answer)

            # 12. Display document sources
            print("\nSources:")

            seen_sources = set()

            for doc in docs:
                source = doc.metadata.get(
                    "source", "Unknown document"
                )
                page = doc.metadata.get("page")

                # Convert zero-based PDF page index to page number
                page_number = (
                    page + 1 if isinstance(page, int) else "Unknown"
                )

                source_label = f"{source}, page {page_number}"

                if source_label not in seen_sources:
                    print(f"- {source_label}")
                    seen_sources.add(source_label)

        except KeyboardInterrupt:
            print("\nChatbot stopped.")
            break

        except Exception as error:
            print(f"\nAn error occurred: {error}")


if __name__ == "__main__":
    main()
