# indexer.py
from dotenv import load_dotenv
load_dotenv()

# ✅ dùng local embeddings để không tốn quota
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

def main():
    # Chỉ index nếu DB đang rỗng
    existing = vector_store.get()
    num_docs = len(existing.get("documents", [])) if existing else 0
    if num_docs > 0:
        print(f"[indexer] Đã có index sẵn ({num_docs} chunks). Không làm gì thêm.")
        return

    print("[indexer] Bắt đầu load & chunk dữ liệu…")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    print(f"[indexer] Thêm {len(all_splits)} chunks vào Chroma…")
    vector_store.add_documents(documents=all_splits)
    print("[indexer] Hoàn tất. Index đã được lưu vào ./chroma_langchain_db")

if __name__ == "__main__":
    main()
