# Manh's Chatbot

A Chainlit-powered chatbot application using LangChain, Google Generative AI, and Chroma vector store. This project demonstrates retrieval-augmented generation (RAG) with local embeddings and streaming responses.

## Features

- Chatbot interface powered by [Chainlit](https://chainlit.io)
- Uses Google Gemini (Generative AI) for chat responses
- Retrieval-augmented generation with Chroma vector store
- Local embeddings via FastEmbed (BAAI/bge-small-en-v1.5)
- Document loading and chunking for context-aware answers
- Multi-language support (see `.chainlit/translations/`)
- Easy environment setup with `.env` and `requirements.txt`

## Getting Started

### Prerequisites

- Python 3.10+
- [Google API Key](https://aistudio.google.com/app/apikey)
- Install dependencies:

```sh
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_actual_google_api_key
```

### Running the App

```sh
chainlit run app.py
```

Open your browser at [http://localhost:8000](http://localhost:8000) to chat with the bot.

## Indexing Documents

To index new documents for retrieval, run:

```sh
python indexer.py
```

## Project Structure

- `app.py` — Main chatbot logic
- `indexer.py` — Document loader and vector store indexer
- `.chainlit/` — Chainlit config and translations
- `chroma_langchain_db/` — Chroma vector store data
- `requirements.txt` — Python dependencies

