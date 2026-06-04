from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.config import settings
import time




def ingest_to_pinecone(pdf_path: str):
    # Load and Split
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Text split strategy
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # Initialize embeddings engine (768 Dimension)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=settings.GOOGLE_API_KEY,
        api_version="v1",
        output_dimensionality=768,
        chunk_size=50,
        retry_min_seconds=10,
    )

    index_name = "ai-engineer"
    batch_size = 25
    total_batches = -(-len(chunks) // batch_size)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=index_name,
        )
        print(f"Upserted batch {batch_num}/{total_batches} ({len(batch)} chunks)")
        if batch_num < total_batches:
            time.sleep(20)

    print(f"Ingested {len(chunks)} chunks from {pdf_path}")


TARGET_PDF = "/Users/martin/Desktop/ai-engineer/ai-projects/projects/stock_rag/docs/f25_AAPL.pdf"

if __name__ == "__main__":
    ingest_to_pinecone(TARGET_PDF)
