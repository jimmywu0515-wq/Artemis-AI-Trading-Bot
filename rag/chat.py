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

If the user wants to change a setting (like buffer percent or strategy), include a tag at the end of your response:
[COMMAND: {{"type": "buffer", "value": 1.5}}] or [COMMAND: {{"type": "toggle_ma", "value": true}}]

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
        # Simple local command parser (works even if offline)
        cmd_tag = ""
        import re
        
        # Buffer pattern: "set buffer to 2.5", "buffer 1.0%"
        buffer_match = re.search(r"buffer.*?(\d+\.?\d*)", query.lower())
        if buffer_match:
            val = float(buffer_match.group(1))
            cmd_tag = f" [COMMAND: {{\"type\": \"buffer\", \"value\": {val}}}]"
            
        # MA Toggle pattern: "show ma", "hide ma", "toggle ma"
        if "show ma" in query.lower() or "enable ma" in query.lower():
            cmd_tag += f" [COMMAND: {{\"type\": \"toggle_ma\", \"value\": true}}]"
        elif "hide ma" in query.lower() or "disable ma" in query.lower():
            cmd_tag += f" [COMMAND: {{\"type\": \"toggle_ma\", \"value\": false}}]"

        if not self.chain:
            # Fallback if no LLM/API KEY
            response = "Knowledge base or LLM API Key is currently offline."
            if "ppo" in query.lower():
                response = "I'm offline, but I utilize PPO models with customized Sharpe Ratio penalizations."
            elif buffer_match or "ma" in query.lower():
                response = "Understood. I've updated the dashboard settings as requested."
            
            return response + cmd_tag
            
        try:
             response = self.chain.invoke(query)
             return response
        except Exception as e:
             print(f"Chat error: {e}")
             return "I encountered an error accessing my memory."
