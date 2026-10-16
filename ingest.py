import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCUMENTS_DIR = "documents"
DB_DIR = "chroma_db"

def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Add GOOGLE_API_KEY to your .env file.")

    pdf_files = [
        os.path.join(DOCUMENTS_DIR, filename)
        for filename in os.listdir(DOCUMENTS_DIR)
        if filename.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise RuntimeError("Add at least one PDF to the documents folder.")

    documents = []
    for pdf_file in pdf_files:
        documents.extend(PyPDFLoader(pdf_file).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2"
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="sample_policies",
    )

    print(f"Indexed {len(chunks)} chunks from {len(pdf_files)} PDF(s).")

if __name__ == "__main__":
    main()