import os

import gradio as gr
try:
    from langchain.chains import RetrievalQA
except Exception:
    from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS


DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""


def set_custom_prompt(custom_prompt_template: str) -> PromptTemplate:
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )


def get_vectorstore() -> FAISS:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def load_llm() -> HuggingFaceEndpoint:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN is missing. Add it in your environment or Hugging Face Space Secrets."
        )

    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.3,
        model_kwargs={
            "token": hf_token,
            "max_length": "512",
        },
    )


def build_qa_chain() -> RetrievalQA:
    vectorstore = get_vectorstore()
    return RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
    )


qa_chain = None


def chat_response(message: str, history) -> str:
    global qa_chain
    if not message or not message.strip():
        return "Please enter a question."
    try:
        if qa_chain is None:
            qa_chain = build_qa_chain()
    except Exception as exc:
        return f"Initialization error: {exc}"

    response = qa_chain.invoke({"query": message})
    return response["result"]


demo = gr.ChatInterface(
    fn=chat_response,
    title="MediBot - Medical Chatbot",
    description="Ask questions from the indexed medical PDFs.",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
