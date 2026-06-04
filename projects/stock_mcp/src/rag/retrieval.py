from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.core.config import settings

# Reinitialize matching embedding engine
embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=settings.GOOGLE_API_KEY,
        output_dimensionality=768
        )
index_name = "ai-engineer"

# Connect directly back to active cloud index instance
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

