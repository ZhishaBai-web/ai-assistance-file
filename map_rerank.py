from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser # 引入 JSON 解析器
from pydantic import BaseModel, Field

def maprerank_response(API,URL,retriever,user_question,character):

    retriever = retriever

    class RerankOutput(BaseModel):
        answer: str = Field(description="问题的回答")
        score: int = Field(description="回答与问题的相关性评分，0-10分")

    parser = JsonOutputParser(pydantic_object=RerankOutput)

    llm = ChatOpenAI(temperature=0.7,
        model="deepseek-v3",
        openai_api_key=API,
        base_url=URL
    )
    if character=="朋友":
        system_message="你是一个喜欢冷嘲热讽，有点傲娇，但心地善良的助手"
    else:system_message="你是一个专业的助手"

    rerank_messages=[
        ("system", system_message),
        ("human", """请依据以下上下文回答问题，并为你的回答打分（0-10分），如果上下文无法回答问题，分数请给0。
                上下文：{context}
                问题：{question}
                 输出格式必须为 JSON：{{"answer": "你的回答内容", "score": 评分数字}}
                """)
        ]
    rerank_prompt = ChatPromptTemplate.from_messages(rerank_messages)

    def map_documents(input_dict):
        docs = input_dict["docs"]
        question = input_dict["question"]
        return [{"context": doc.page_content, "question": question} for doc in docs]

    def pick_best_answer(answers):
        best = max(answers, key=lambda x: x.get("score", 0))
        return best["answer"]

    single_unit_chain =  rerank_prompt | llm | parser

    reduce_chain = ({"docs": retriever, "question": RunnablePassthrough()} |
                    RunnableLambda(map_documents) | single_unit_chain.map() |
                    RunnableLambda(pick_best_answer))
    final_response = reduce_chain.invoke(user_question)

    return final_response