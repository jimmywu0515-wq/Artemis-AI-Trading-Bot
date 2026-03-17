import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "docs")

def build_knowledge_base():
    """
    Reads markdown/text files from the rag/docs directory, splits them into searchable chunks,
    and returns a retriever interface over a FAISS vectorstore.
    """
    
    # Initialize basic directory if it doesn't exist
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        with open(os.path.join(KNOWLEDGE_DIR, "strategy.md"), "w") as f:
            f.write("# RL Grid Strategy\nWe use a Proximal Policy Optimization (PPO) agent from Stable-Baselines3. The bot attempts to adjust grid centers and boundaries by penalizing Maximum Drawdown.")
            
    try:
        loader = DirectoryLoader(KNOWLEDGE_DIR, glob="**/*.md", loader_cls=TextLoader)
        docs = loader.load()
        
        if not docs:
            print("No documents found in knowledge base.")
            return None
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        # NOTE: Requires OPENAI_API_KEY
        embeddings = OpenAIEmbeddings()
        
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        return vectorstore.as_retriever()
    except Exception as e:
        print(f"Failed to build vectorstore: {e}")
        return None
