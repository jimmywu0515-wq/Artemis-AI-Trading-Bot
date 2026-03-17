import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .knowledge_base import build_knowledge_base

class RAGChatbot:
    def __init__(self):
        self.retriever = None
        self.chain = None
        self._initialize()
        
    def _initialize(self):
        if not os.getenv("OPENAI_API_KEY"):
            print("Chatbot Offline: OPENAI_API_KEY missing.")
            return
            
        self.retriever = build_knowledge_base()
        
        if self.retriever:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            
            template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer or the context doesn't contain it, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
            
            custom_rag_prompt = PromptTemplate.from_template(template)
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
                
            self.chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | custom_rag_prompt
                | llm
                | StrOutputParser()
            )
            
    def ask(self, query: str) -> str:
        if not self.chain:
            # Fallback if no LLM/API KEY
            if "ppo" in query.lower():
                return "I'm offline, but I utilize PPO models with customized Sharpe Ratio penalizations for deep-drawdowns."
            return "Knowledge base or LLM API Key is currently offline."
            
        try:
             response = self.chain.invoke(query)
             return response
        except Exception as e:
             print(f"Chat error: {e}")
             return "I encountered an error accessing my memory."
