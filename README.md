# RAG Document Chatbot

A Python-based Retrieval-Augmented Generation (RAG) chatbot that answers questions using information retrieved from PDF documents.

The project uses LangChain for the RAG workflow, ChromaDB for vector storage, and Google Gemini for embeddings and answer generation.

## Features

- Load and process PDF documents.
- Split document content into smaller text chunks.
- Generate embeddings for semantic search.
- Store and retrieve document vectors using ChromaDB.
- Answer questions using relevant document context.
- Display source document names and page numbers.
- Avoid answering from unsupported information when the answer is not found in the supplied context.

## Tech Stack

- Python
- LangChain
- Google Gemini API
- ChromaDB
- PyPDF
- python-dotenv

## Project Structure

```text
RAG/
├── documents/
│   └── sample-rag.pdf
├── chroma_db/
├── .venv/
├── .env
├── ingest.py
├── chat.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Prerequisites

- Python 3.10 or a compatible version supported by the installed dependencies.
- Git (optional).
- A Google Gemini API key from https://aistudio.google.com/apikey.

## Setup and Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd RAG
```

Skip this step if the project is already on your computer.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run Python using `.venv\Scripts\python.exe` directly.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Replace the placeholder with your actual API key.

**Security:** Never commit `.env` or expose your API key in source code.

### 5. Add PDF documents

Place your PDF files inside the `documents/` directory.

Ensure the files are valid PDFs and contain extractable text.

### 6. Build the vector database

Run the ingestion script:

```powershell
python ingest.py
```

This loads the PDFs, splits their text into chunks, generates embeddings, and stores the vectors in ChromaDB.

Run ingestion again whenever you add or update documents, according to the behavior implemented in `ingest.py`.

### 7. Start the chatbot

```powershell
python chat.py
```

Ask questions in the terminal. Type `exit` or `quit` to stop the chatbot.

## Example

**Question:**

What is the collision deductible?

**Example answer:**

The collision deductible is $500.

The actual answer depends on the content of the PDF documents available to the application.

## How the RAG Pipeline Works

1. **Document loading:** Extract text from PDFs.
2. **Chunking:** Split documents into smaller sections.
3. **Embedding:** Convert text chunks into numerical vectors.
4. **Vector storage:** Store embeddings and associated text in ChromaDB.
5. **Retrieval:** Search for relevant chunks when a question is asked.
6. **Augmentation:** Include retrieved text in the language model prompt.
7. **Generation:** Use Gemini to generate an answer based on the supplied context.

## Troubleshooting

- **Missing API key:** Verify that `.env` exists in the project root and contains `GOOGLE_API_KEY`.
- **PDF parsing error:** Confirm that the document is a valid PDF, not a text file renamed with a `.pdf` extension.
- **Empty or irrelevant results:** Run `ingest.py` and verify that the correct ChromaDB collection and embedding model are used.
- **Module not found:** Activate the correct virtual environment and reinstall dependencies.
- **API or quota error:** Check Gemini API access, model availability, and usage limits.

## Important Notes

- Keep the embedding model and ChromaDB collection configuration consistent between ingestion and chat.
- Do not commit API keys, virtual environments, or confidential documents.
- Use only documents you are authorized to process.
- Answers may be incorrect or incomplete; verify important information against the original documents.

## Future Enhancements

- FastAPI backend and React frontend.
- PDF upload functionality.
- Conversation history.
- Streaming responses.
- Docker-based deployment.
- Authentication and persistent cloud vector storage.

## License

Add a license before distributing or reusing this project publicly.
