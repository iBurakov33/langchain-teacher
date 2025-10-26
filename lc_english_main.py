import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from get_prompt import load_prompt_solve_task, load_prompt_make_task
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.chat_models import ChatOllama


st.set_page_config(page_title="Интерактивные уроки на английском языке")
st.title("Интерактивные уроки на английском языке")
button_css = """.stButton>button {
    color: #4F8BF9;
    border-radius: 50%;
    height: 2em;
    width: 2em;
    font-size: 4px;
}"""
st.markdown(f'<style>{button_css}</style>', unsafe_allow_html=True)

MODEL = "llama3"


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# Lesson selection dictionary
lesson_guides = {
    "Тема 1: Активный и Пассивный залог": {
        "file": "lessons/passive_voice.txt",
        "description": "В данной теме рассматривается то, как использовать в предложениях активный и пассивный залог: \
        правила и примеры с active voice и passive voice."
    },
    "Тема 2: Формальные подлежащие": {
        "file": "lessons/it_clause.txt",
        "description": "В данной теме рассматривается формальное подлежащее it, правила и примеры использования."
    },
}

# Lesson selection sidebar
lesson_selection = st.sidebar.selectbox("Выбрать тему", list(lesson_guides.keys()))

# Display lesson content and description based on selection
lesson_info = lesson_guides[lesson_selection]
lesson_content = open(lesson_info["file"], "r").read()
lesson_description = lesson_info["description"]

# Radio buttons for lesson type selection
lesson_type = st.sidebar.radio("Выбрать тип задания", ["Решение задания", "Генерация задания"])

# Clear chat session if dropdown option or radio button changes
if st.session_state.get("current_lesson") != lesson_selection or st.session_state.get(
        "current_lesson_type") != lesson_type:
    st.session_state["current_lesson"] = lesson_selection
    st.session_state["current_lesson_type"] = lesson_type
    st.session_state["messages"] = [AIMessage(
        content="Добро пожаловать! Выберете тему и тип задания для беседы. Дайте мне знать, когда будете готовы!")]

# Display lesson name and description
st.markdown(f"**{lesson_selection}**")
st.write(lesson_description)

for msg in st.session_state["messages"]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        model = ChatOllama(streaming=True, callbacks=[stream_handler], model=MODEL)

        if lesson_type == "Решение задания":
            prompt_template = load_prompt_solve_task(content=lesson_content)
        else:
            prompt_template = load_prompt_make_task(content=lesson_content)

        chain = prompt_template | model | StrOutputParser()
        #print(prompt_template.invoke(
        #    input={"input": prompt, "chat_history": st.session_state.messages[-20:]},
        #))

        response = chain.invoke(
            input={"input": prompt, "chat_history": st.session_state.messages[-20:]},
        )

        st.session_state.messages.append(HumanMessage(content=prompt))
        st.session_state.messages.append(AIMessage(content=response))
