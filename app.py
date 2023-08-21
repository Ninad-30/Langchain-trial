# Dependencies

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

def get_pdf_txt(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text  += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)
    
    return text_splitter.split_text(raw_text)


def get_vector_store(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vector_store = FAISS.from_texts(texts=text_chunks, embeddings=embeddings)
    return vector_store





def main():
    load_dotenv()

    # Load the Streamlit website GUI
    st.set_page_config(page_title="Research paper chatter", page_icon=":books:")
    st.header("Research Paper summarizer :books:")
    st.text_input("Ask a question about the paper")

    with st.sidebar:
        st.subheader("Your research papers")
        pdf_docs = st.file_uploader("Upload the PDFs here",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):    
                # Process the PDFs once the button is pressed
                raw_text = get_pdf_txt(pdf_docs)
                st.write("PDF uploaded successfully")

                # Text chunks
                text_chunks = get_text_chunks(raw_text)

                # Vector db
                vectorstore = get_vector_store(text_chunks)






if __name__ == "__main__":
    main()

