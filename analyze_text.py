from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader

from get_prompt import load_prompt_analyze_text
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.chat_models import ChatOllama
import os
import spacy
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('sentencizer')


MODEL = "llama3"

lesson_guides = {
    "Тема 1: Активный и Пассивный залог": {
        "file": "lessons/passive_voice3.txt",
        "description": "В данной теме рассматривается то, как использовать в предложениях активный и пассивный залог: \
        правила и примеры с active voice и passive voice."
    },
    "Тема 2: Формальные подлежащие": {
        "file": "lessons/it_clause.txt",
        "description": "В данной теме рассматривается формальное подлежащее it, правила и примеры использования."
    },
}

lesson_selection = list(lesson_guides.keys())
lesson_info = lesson_guides[lesson_selection[0]]
lesson_content = open(lesson_info["file"], "r").read()
lesson_description = lesson_info["description"]

model = ChatOllama(streaming=True, model=MODEL)

#prompt_template = load_prompt_analyze_text(content="")

#chain = prompt_template | model | StrOutputParser()
chain = model | StrOutputParser()

model_list = [
    "llama3",
    "gemma3",
    "deepseek-r1",
    "mistral",
    "qwen2.5",
    "neural-chat",
    "starling-lm",
    "owl/t-lite"
]
for MODEL in model_list:
    #MODEL = "llama3"
    save_dir = f"data/{MODEL}/by_sentence_agentic2"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    lesson_selection = list(lesson_guides.keys())
    lesson_info = lesson_guides[lesson_selection[0]]
    lesson_content = open(lesson_info["file"], "r").read()
    lesson_description = lesson_info["description"]

    model = ChatOllama(streaming=True, model=MODEL, temperature=0.0)

    #prompt_template = load_prompt_analyze_text(content="")

    #chain = prompt_template | model | StrOutputParser()
    chain = model | StrOutputParser()

    #lesson_prompt = f"""This is an explanation of the passive voice in English grammar:
    #                                {lesson_content}
    #                                Make a detailed explanation on how to find passive voice in a text.
    #                                """
    #lesson_response = chain.invoke(lesson_prompt)

    articles_dir = "data/articles"
    for path, directories, files in os.walk(articles_dir):
        for file in files:
            file_txt = file.replace('.pdf', '.txt')
            with open(f"{save_dir}/{file_txt}", 'a', encoding="utf-8") as f:
                #article = f"{articles_dir}/2The ocean is as important to the climate as the atmosphere.pdf"
                loader = PyPDFLoader(os.path.join(path, file))
                data = loader.load()

                text = ''
                for page_data in data:
                    text += page_data.page_content.replace("\n", "").replace("  ", " ")
                tokens = nlp(text)
                for sent in tokens.sents:
                    #print(sent.text.strip())

                    #prompt = f"""Determine whether the input sentence contains passive voice.
                    #            {sent.text.strip()}
                    #            Output must only be 'Passive voice' if the sentence uses passive voice.
                    #            If the sentence does not use passive voice output 'Active voice'.
                    #            Output nothing else.
                    #            """

                    #prompt = f"""This is an explanation of the passive voice in English grammar:
                    #             {lesson_content}
                    #             Using the explanation determine whether the following sentence is in passive voice.
                    #             {sent.text.strip()}
                    #             Output must only be 'Passive voice' if the sentence uses passive voice.
                    #             If the sentence does not use passive voice output 'Active voice'.
                    #             Output nothing else.
                    #             """


                    prompt = f"""This is an explanation of how to find passive voice in a text:
{lesson_content}
Using the explanation determine whether the following sentence is in passive voice.
{sent.text.strip()}
"""
                    response = chain.invoke(prompt)

                    agentic_prompt = f"""This is an explanation of how to find passive voice in a text:
{lesson_content}
Use this explanation to correct any mistakes in your previous response and determine whether the sentence is in passive voice.
Previous response:
{response}
Output must only be 'Passive voice' if the sentence uses passive voice.
If the sentence does not use passive voice output 'Active voice'.
Output nothing else.
"""
                    agentic_response = chain.invoke(agentic_prompt)
                    #used_voice = 'Passive voice' if 'Passive voice' in response else 'Active voice'
                    print(f"{sent.text.strip()} - {agentic_response}")
                    print('--------------------------------------')
                    if 'Passive voice' in agentic_response:
                        f.write(f"{sent.text.strip()}\n")
        f.close()
