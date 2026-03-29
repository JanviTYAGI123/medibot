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
