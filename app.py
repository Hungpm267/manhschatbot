from dotenv import load_dotenv
load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI
# pip install langchain-google-genai
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast

import chainlit as cl

from langchain.chat_models import init_chat_model
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", streaming = True)

# ⛔ Bỏ:
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# ✅ Dùng local: Dùng embedding local (miễn phí). pip install sentence-transformers langchain-community
# from langchain_community.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


from langchain_chroma import Chroma
# vector_store = Chroma(
#     collection_name="example_collection",
#     embedding_function=embeddings,
#     persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
# )

from langchain_community.embeddings import FastEmbedEmbeddings
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict



from langgraph.graph import MessagesState, StateGraph
# graph_builder = StateGraph(MessagesState)

from langchain_core.tools import tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}
# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])
# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. if the question not related to the document, say you refuse to answer. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

def build_graph():
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}

# while True:
#   input_message = input("user: ")
#   if input_message == "exit":
#       break
#   for step in graph.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode="values",
#     config=config,
#   ):
#       step["messages"][-1].pretty_print()

@cl.on_chat_start
async def on_chat_start():
    # await cl.Message(content="Xin chào. rất vui được gặp bạn").send()
    thread_id = "abc123"
    graph = build_graph()
    cl.user_session.set("graph", graph)
    cl.user_session.set("thread_id", thread_id)
    # model = ChatGoogleGenerativeAI(
    # model="gemini-2.5-flash",
    # temperature=0.7,
    # streaming=True,
    # convert_system_message_to_human=True  # để tương thích prompt dạng "system"
    # )

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
    #         ),
    #         ("human", "{question}"),
    #     ]
    # )
    # runnable = prompt | model | StrOutputParser()
    # cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(input_message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    graph = cl.user_session.get("graph")

    msg = cl.Message(content="")

    # Duyệt qua các sự kiện từ graph
    async for event in graph.astream_events(
        {"messages": [{"role": "user", "content": input_message.content}]},
        config={"configurable": {"thread_id": thread_id}},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            await msg.stream_token(chunk)

    await msg.send()  # gửi tin nhắn khi xong

    #=====================================================================
    # # Gọi graph để lấy kết quả hoàn chỉnh
    # result = await graph.ainvoke(
    #     {"messages": [{"role": "user", "content": input_message.content}]},
    #     config={"configurable": {"thread_id": thread_id}},
    # )

    # # Lấy AI message cuối cùng
    # last = result["messages"][-1]
    # # Nếu chắc chắn last.content là string:
    # msg.content = last.content
    # # Gửi một lần
    # await msg.send()

    #============================================================================
    # runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    # msg = cl.Message(content="")

    # async for chunk in runnable.astream(
    #     {"question": message.content},
    #     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    # ):
    #     await msg.stream_token(chunk)

    # await msg.send()

