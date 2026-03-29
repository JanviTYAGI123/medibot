🔹 Project: MediBot – AI Medical Chatbot (RAG-based)
Built an end-to-end Retrieval-Augmented Generation (RAG) based medical chatbot using LangChain, FAISS, and HuggingFace (Mistral-7B-Instruct)
Designed a vector database pipeline by processing medical PDFs, chunking data, and generating embeddings using Sentence Transformers
Implemented semantic search with FAISS, improving response relevance and reducing hallucinations in LLM outputs
Developed a modular RAG pipeline integrating LLM with retrieval system for context-aware responses
Engineered custom prompt templates to ensure medically safe outputs with disclaimers
Built an interactive chatbot UI using Gradio, enabling real-time query handling and chat history
Optimized performance by caching vector store and reducing redundant computations
Handled edge cases with input validation, error handling, and user-friendly feedback
Designed scalable architecture with clear separation of concerns (UI, pipeline, vector DB)
Deployed the application on HuggingFace Spaces for public access



# Medical Chatbot (Gradio + Hugging Face Spaces)

This project is now configured to run with **Gradio** using `app.py`, ready to deploy on Hugging Face Spaces.

## Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Hugging Face token in environment:
```bash
set HF_TOKEN=your_hf_token_here
```

3. Start the app:
```bash
python app.py
```

## Deploy on Hugging Face Spaces

1. Create a new **Space** with SDK: **Gradio**.
2. Upload this project files (including `app.py`, `requirements.txt`, `vectorstore/`, and `data/` if needed).
3. In Space **Settings -> Variables and secrets**, add:
   - `HF_TOKEN` = your Hugging Face token
4. Deploy/restart the Space. It will auto-run `app.py`.

## Notes

- `app.py` uses FAISS index from `vectorstore/db_faiss`.
- If you update PDFs in `data/`, re-run `create_memory_for_llm.py` to rebuild embeddings.
