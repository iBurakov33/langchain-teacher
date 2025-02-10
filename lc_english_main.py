import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from get_prompt import load_prompt, load_prompt_with_questions

st.set_page_config(page_title="LangChain: Getting Started Class")
st.title("LangChain: Getting Started Class")
button_css = """.stButton>button {
    color: #4F8BF9;
    border-radius: 50%;
    height: 2em;
    width: 2em;
    font-size: 4px;
}"""
st.markdown(f'<style>{button_css}</style>', unsafe_allow_html=True)

MODEL = "deepseek-r1:1.5b"


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# Lesson selection dictionary
lesson_guides = {
    "Lesson 1: Getting Started with LangChain": {
        "file": "lc_guides/getting_started_guide.txt",
        "description": "This lesson covers the basics of getting started with LangChain."
    },
    "Lesson 2: Prompts": {
        "file": "lc_guides/prompt_guide.txt",
        "description": "This lesson focuses on prompts and their usage."
    },
    "Lesson 3: Language Models": {
        "file": "lc_guides/models_guide.txt",
        "description": "This lesson provides an overview of language models."
    },
    "Lesson 4: Memory": {
        "file": "lc_guides/memory_guide.txt",
        "description": "This lesson is about Memory."
    },
    "Lesson 5: Chains": {
        "file": "lc_guides/chains_guide.txt",
        "description": "This lesson provides information on Chains in LangChain, their types, and usage."
    },
    "Lesson 6: Retrieval": {
        "file": "lc_guides/retrieval_guide.txt",
        "description": "This lesson provides information on indexing and retrieving information using LangChain."
    },
    "Lesson 7: Agents": {
        "file": "lc_guides/agents_guide.txt",
        "description": "This lesson provides information on agents, tools, and toolkits."
    }
}

# Lesson selection sidebar
lesson_selection = st.sidebar.selectbox("Select Lesson", list(lesson_guides.keys()))

# Display lesson content and description based on selection
lesson_info = lesson_guides[lesson_selection]
lesson_content = open(lesson_info["file"], "r").read()
lesson_description = lesson_info["description"]

# Radio buttons for lesson type selection
lesson_type = st.sidebar.radio("Select Lesson Type", ["Instructions based lesson", "Interactive lesson with questions"])

# Clear chat session if dropdown option or radio button changes
if st.session_state.get("current_lesson") != lesson_selection or st.session_state.get(
        "current_lesson_type") != lesson_type:
    st.session_state["current_lesson"] = lesson_selection
    st.session_state["current_lesson_type"] = lesson_type
    st.session_state["messages"] = [AIMessage(
        content="Welcome! This short course will help you get started with LangChain. Let me know when you're all set to jump in!")]

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

        if lesson_type == "Instructions based lesson":
            prompt_template = load_prompt(content=lesson_content)
        else:
            prompt_template = load_prompt_with_questions(content=lesson_content)

        chain = prompt_template | model | StrOutputParser()
        #print(prompt_template.invoke(
        #    input={"input": prompt, "chat_history": st.session_state.messages[-20:]},
        #))

        response = chain.invoke(
            input={"input": prompt, "chat_history": st.session_state.messages[-20:]},
        )

        st.session_state.messages.append(HumanMessage(content=prompt))
        st.session_state.messages.append(AIMessage(content=response))
