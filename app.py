# Dependencies

import os
import requests
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

from langchain.vectorstores import FAISS

from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
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


def main():
    load_dotenv()

    # Load the Streamlit website GUI
    st.set_page_config(page_title="Research paper chatter", page_icon=":books:")
    st.header("Research Paper summarizer :books:")
    question = st.text_input("Ask a question about the paper")

    with st.sidebar:
        st.subheader("Your research papers")
        pdf_docs = st.file_uploader("Upload the PDFs here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Process the PDFs once the button is pressed
                raw_text = get_pdf_txt(pdf_docs)
                st.write(raw_text[:10])

                # Text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # Vector db
                vectorstore = get_vector_store(text_chunks)
                print(type(vectorstore))
                print("Stored the pdf into a vector db")

                while not question:
                    st.spinner("Input a question please!")
                docs = vectorstore.similarity_search(question)
                print(docs[0].page_content)


if __name__ == "__main__":
    main()
