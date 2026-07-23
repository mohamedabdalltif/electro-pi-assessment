import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.embeddings import Embeddings

from embedder import Embedder

load_dotenv()


class OnnxEmbeddings(Embeddings):
    """Wraps our ONNX Embedder so LangChain can use it."""

    def __init__(self, model_path="models/Xenova/all-MiniLM-L6-v2"):
        self._embedder = Embedder(path=model_path)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode_batch(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.encode(text).tolist()


def load_documents(docs_dir="./docs"):
    docs = []
    for md_file in Path(docs_dir).glob("*.md"):
        loader = TextLoader(str(md_file), encoding="utf-8")
        docs.extend(loader.load())
    return docs


def build_vectorstore(docs_dir="./docs"):
    print("Loading documents...")
    docs = load_documents(docs_dir)

    print("Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"  {len(chunks)} chunks created from {len(docs)} documents")

    print("Embedding with ONNX (all-MiniLM-L6-v2)...")
    embeddings = OnnxEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("  FAISS index built.")
    return vectorstore


def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model="openai/gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=512,
    )

    prompt = PromptTemplate.from_template(
        """You are a helpful customer support assistant for a food delivery app.
Answer the question using ONLY the context provided below.

Rules:
1. Cite the source document name in your answer (e.g. [refund_policy.md]).
2. If the answer cannot be found in the context, respond EXACTLY with:
   "I cannot answer this based on the provided documents. No relevant context was found."

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        if not docs:
            return "No relevant context found."
        parts = []
        for doc in docs:
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            parts.append(f"[{source}]\n{doc.page_content}")
        return "\n\n".join(parts)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    vs = build_vectorstore()
    chain = build_rag_chain(vs)

    q = "What is the refund policy for missing items?"
    print(f"\nQ: {q}")
    print(f"A: {chain.invoke(q)}")
