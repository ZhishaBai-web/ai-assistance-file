from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import MessagesPlaceholder
from user_retriver import document_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from user_token import count_tokens

store = {}
def stuff_response(API,URL,retriever,user_question,character):
    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    def summarize_messages(session_id):
        history = get_session_history(session_id)
        messages = history.messages

        if count_tokens(messages) > 1500:
            summary_prompt = f"请简要总结以下对话内容的要点：\n{messages}"
            summary_content = llm.invoke(summary_prompt)

            history.clear()
            history.add_ai_message(f"这是之前的对话摘要：{summary_content}")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = retriever
    llm = ChatOpenAI(temperature=0.7,
        model="deepseek-v3",
        openai_api_key=API,
        base_url=URL
    )

    human_message = """优先依据以下上下文内容回答问题,如果用户的问题与上下文内容无关，可直接根据{history}回答
                上下文内容:{context}
                问题：{question}
                """
    if character=="朋友":
        system_message="你是一个喜欢冷嘲热讽，有点傲娇，但心地善良的助手"
    else:system_message="你是一个专业的助手"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="history"),
        ("human", human_message)])

    rag_chain = (
            {"context": (lambda x: x["question"]) | retriever | format_docs,
             "question": (lambda x: x["question"]),
             "history": lambda x: x["history"]}
            | prompt
            | llm
            | StrOutputParser()
    )

    with_message_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    config = {"configurable": {"session_id": "user_01"}}

    response = with_message_history.invoke(
        {"question": user_question},
        config=config
    )
    summarize_messages("user_01")
    return response



