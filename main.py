from user_retriver import document_retriever
from refine import refine_response
from stuff import stuff_response
from map_reduce import mapreduce_response
from map_rerank import maprerank_response
import streamlit as st

def clear_retriever():
    if "retriever" in st.session_state:
        del st.session_state["retriever"]
    if "submit" in st.session_state:
        st.session_state["submit"] = False
    if "messages" in st.session_state:
        del st.session_state["messages"]

def clear_submit():
    if "submit" in st.session_state:
        st.session_state["submit"] = False
    if "messages" in st.session_state:
        del st.session_state["messages"]


st.title("📕 AI智能问答工具")
with st.sidebar:
    API = st.text_input("请输入您调用的API",type="password")
    URL = st.text_input("请输入您调用的URL", type="password")
    st.markdown("[您可在该API中转站，获取API和URL](https://aigc789.top/)")

upload_file=st.file_uploader("上传需要问答的PDF文件",type="pdf", on_change=clear_retriever)

column1,column2,column3=st.columns([1,1,1])
with column1:
    user_language = st.selectbox("上传文件的语言",["中文","English","其他"],
                                 index=None,on_change=clear_retriever)
with column2:
    respond_model = st.selectbox("解析文件的方法",["stuff","Refine","Map-Reduce","Map-Rerank"],
                                 index=None,on_change=clear_submit)
with column3:
    character=st.selectbox("您对AI的看法",["工具","朋友","其他"],index=None,on_change=clear_submit)

with st.expander("一些注意事项"):
    st.markdown("""  1. 上传文件语言的选择，会影响分割器对段落的划分。  
    2. stuff方法，适用长度较短的文档。token消耗较少，反应快，存在记忆。
    3. Map-Reduce方法，适用超长文档总结、大规模数据筛选。token消耗多，反应较快。
    4. Refine方法，适用需要深度理解、对结果质量要求极高的任务。token消耗多，反应慢。  
    5. Map-Rerank方法，适用答案隐藏于某一个特定段落的问答。token消耗较多，反应较快。  
    6. 对AI看法的选择，将细微影响AI的性格。  
    7. 调用的模型 deepseek-v3 。  """)


col1,col2,col3=st.columns([1.3,1,1])
with col2:
    submit=st.button("基础设置已完成")

if "submit" not in st.session_state:
    st.session_state["submit"] = False
if submit:
    st.session_state["submit"] = True


if not API:
    st.info("请输入您的API")
    st.stop()
if not URL:
    st.info("请输入您的URL")
    st.stop()
if not upload_file:
    st.info("请上传您的文件")
    st.stop()
if not user_language:
    st.info("请选择文件的语言")
    st.stop()
if not respond_model:
    st.info("请选择解析文件的方法")
    st.stop()
if not character:
    st.info("请选择对AI的看法")
    st.stop()
if not st.session_state["submit"]:
    st.info('请点击"基础设置已完成"按钮')
    st.stop()

if "submit" not in st.session_state:
        st.session_state["submit"] = True

if "retriever" not in st.session_state:
        st.session_state["retriever"] = document_retriever(upload_file,user_language,API,URL)


if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"ai",
                                  "content":"你好，我是AI助手"}]

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

question=st.chat_input()

if question:
    st.session_state["messages"].append({"role":"human",
                                        "content":question})
    st.chat_message("human").write(question)

    with st.spinner("AI正在思考，请稍等..."):
        if respond_model == "stuff":
            response = stuff_response(API, URL, st.session_state["retriever"], question, character)
        if respond_model == "Refine":
            response = refine_response(API, URL, st.session_state["retriever"], question, character)
        if respond_model == "Map-Reduce":
            response = mapreduce_response(API, URL, st.session_state["retriever"], question, character)
        if respond_model == "Map-Rerank":
            response = maprerank_response(API, URL, st.session_state["retriever"], question, character)

    msg={"role":"ai","content":response}
    st.session_state["messages"].append(msg)
    st.chat_message("ai").write(response)




