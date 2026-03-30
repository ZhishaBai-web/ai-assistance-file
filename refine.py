from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import MessagesPlaceholder


def refine_response(API,URL,retriever,user_question,character):

    retriever = retriever
    llm = ChatOpenAI(temperature=0.7,
        model="deepseek-v3",
        openai_api_key=API,
        base_url=URL
    )
    if character=="朋友":
        system_message="你是一个喜欢冷嘲热讽，有点傲娇，但心地善良的助手"
    else:system_message="你是一个专业的助手"

    initial_messages=[
        ("system", system_message),
        ("human", """请根据以下内容，开始回答问题：
                    内容：{context}
                    问题：{question}
                    初步回答：""")
        ]
    initial_prompt = ChatPromptTemplate.from_messages(initial_messages)
    initial_prompt_chain=initial_prompt | llm | StrOutputParser()

    refine_template = [
        ("system", system_message),
        ("human","""你的任务是产生一个最终的回答。
            我们已经有了一个初步的回答：{existing_answer}
            现在我们有一段新的参考内容：{context}
            请结合这段新内容，改进或完善之前的回答。如果新内容没有帮助，请保留原样。
            问题：{question}
            完善后的回答：""")
    ]
    refine_prompt = ChatPromptTemplate.from_messages(refine_template)
    refine_prompt_chain=refine_prompt | llm | StrOutputParser()


    def run_refine_logic(inputs):
        docs = inputs["docs"]
        question = inputs["question"]

        first_doc_content = docs[0].page_content
        current_answer = initial_prompt_chain.invoke({"context": first_doc_content, "question": question})

        for i in range(1, len(docs)):
            current_answer = refine_prompt_chain.invoke(
                {"existing_answer": current_answer, "context": docs[i].page_content, "question": question})

        return current_answer

    refine_chain = (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(run_refine_logic)
    )

    final_response = refine_chain.invoke(user_question)
    return final_response

