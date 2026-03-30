from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import MessagesPlaceholder

def mapreduce_response(API,URL,retriever,user_question,character):

    retriever = retriever
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=API,
        base_url=URL
    )
    if character=="朋友":
        system_message="你是一个喜欢冷嘲热讽，有点傲娇，但心地善良的助手"
    else:system_message="你是一个专业的助手"

    map_messages=[
        ("system", system_message),
        ("human", """根据以下文档片段回答问题：
               文档片段：{context}
               问题：{question}
               局部回答：     """)
        ]
    map_prompt = ChatPromptTemplate.from_messages(map_messages)
    map_chain=map_prompt | llm | StrOutputParser()

    reduce_template = [
        ("system", system_message),
        ("human", """以下是针对同一个问题的多个局部回答：
                    {summaries}
                    请综合以上内容，给出一个最终的完整答案：""")
        ]
    reduce_prompt = ChatPromptTemplate.from_messages(reduce_template)
    reduce_chain=reduce_prompt | llm | StrOutputParser()


    def map_reduce_process(inputs):
        docs = retriever.invoke(inputs)
        summaries = [map_chain.invoke({"context": doc.page_content, "question": inputs})
                     for doc in docs]
        final_answer = reduce_chain.invoke({"summaries": "\n\n".join(summaries)})
        return final_answer

    map_reduce_chain = RunnableLambda(map_reduce_process)
    final_response = map_reduce_chain.invoke(user_question)

    return final_response