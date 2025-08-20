from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
import textwrap

INDEX_DIR = "faiss_index"
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "mistral"

if not os.path.isdir(INDEX_DIR):
    raise RuntimeError(f"Vector index not found at {INDEX_DIR}. Run ingest.py first.")

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

@tool("knowledge_search", return_direct=False)
def knowledge_search(query: str) -> str:
    """Use this to search the local knowledge base for facts relevant to the user question."""
    docs = retriever.get_relevant_documents(query)
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "local")
        parts.append(f"[{i}] (source: {os.path.basename(src)})\n{d.page_content}")
    return "\n\n".join(parts) if parts else "No relevant documents found."

AGENT_PROMPT = ChatPromptTemplate.from_template(textwrap.dedent("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
"""))

llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0,
    top_p=0.9,
    repeat_penalty=1.1,
    num_predict=512,
)

tools = [knowledge_search]
react_agent = create_react_agent(llm, tools, AGENT_PROMPT)
agent = AgentExecutor(agent=react_agent, tools=tools, verbose=False) 

RAG_PROMPT = ChatPromptTemplate.from_template(textwrap.dedent("""
You are a helpful assistant. Use the provided context to answer the user's question.
If the answer cannot be found in the context, say you don't know.

Context:
{context}

Question:
{question}
"""))

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

def main():
    print("To quit from chatbot type: exit")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            break

        try:
            result = agent.invoke({"input": user_input})
            answer = result["output"]
        except Exception as e:
            answer = rag_chain.invoke(user_input)

        print("\nBot:", answer)

if __name__ == "__main__":
    main()

