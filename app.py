# Dependencies

import os
import requests
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

from langchain.vectorstores import FAISS

from langchain.llms import HuggingFaceHub, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

from apikeys import *

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN


def get_pdf_txt(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)

    return text_splitter.split_text(raw_text)


# def get_vector_store(text_chunks):
# model_id = 'hkunlp/instructor-xl'
# model_id = "sentence-transformers/all-MiniLM-L6-v2"
# api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
# headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACEHUB_API_TOKEN']}"}

# print("Initiating request")
# response = requests.post(api_url, headers=headers, json={"inputs": text_chunks, "options":{"wait_for_model":True}})
# print("Response recieved")
# return response.json()


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain


def handle_userinput(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 ==0:
            st.write(message.content)
        else:
            st.write(message.content)


def main():
    load_dotenv()

    # Load the Streamlit website GUI
    st.set_page_config(page_title="Research paper chatter", page_icon=":books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Research Paper chat :books:")
    user_question = st.text_input("Input the PDF(s), press the Process button, and ask a question when prompted!")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your research papers")
        pdf_docs = st.file_uploader("Upload the PDFs here and click to Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Process the PDFs once the button is pressed
                raw_text = get_pdf_txt(pdf_docs)
                st.write(raw_text[:20])

                # Text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # Vector db
                vectorstore = get_vector_store(text_chunks)
                print("Stored the pdf into a vector db")

                # docs = vectorstore.similarity_search(question)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.write("Ask a question now")


if __name__ == "__main__":
    main()
