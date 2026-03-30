import tiktoken
from langchain_core.messages import get_buffer_string
from langchain_core.messages import HumanMessage, AIMessage


def count_tokens(messages):
    encoding = tiktoken.get_encoding("cl100k_base")
    text = get_buffer_string(messages)

    return len(encoding.encode(text))

# messages = [
#     HumanMessage(content="你好，请问苯-异丙醇-水的分离效率如何？"),
#     AIMessage(content="这是一个关于三元共沸物分离的问题...")
# ]
# long=count_tokens(messages)
# print(long)