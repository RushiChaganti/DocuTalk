import streamlit as st
import fitz 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import time

# Set page configuration
st.set_page_config(
    page_title="ChatPDF",
    page_icon=":clipboard:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None,
    },
)

# Initialize session state
if 'uploaded_pdfs' not in st.session_state:
    st.session_state.uploaded_pdfs = []

# Function to load document and QA
def load_doc_and_qa(pdf_doc):
    try:
        pdf_bytes = pdf_doc.read()

        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = [page.get_text() for page in pdf_document]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        text = text_splitter.create_documents(text)
        embedding = HuggingFaceEmbeddings()
        db = Chroma.from_documents(text, embedding)
      
        llm = HuggingFaceHub(repo_id="google/flan-ul2", model_kwargs={"temperature": 1.0, "max_length": 256})
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

        st.session_state.uploaded_pdfs.append((pdf_doc, chain))

        return 'Document has been successfully loaded'
    except Exception as e:
        st.error(f"Error loading the document: {str(e)}")

# Streamlit UI
st.title("ChatPDF")
st.write("Upload PDF Files. Once uploaded, you can begin chatting with each PDF :)")

pdf_docs = st.file_uploader("Load PDF files", type=["pdf"], accept_multiple_files=True)

if pdf_docs:
    st.write("Processing PDFs...")
    for pdf_doc in pdf_docs:
        with st.spinner(f"Loading PDF: {pdf_doc.name}..."):
            load_doc_and_qa(pdf_doc)
            st.success(f"PDF '{pdf_doc.name}' loaded successfully")

for pdf_doc, qa_instance in st.session_state.uploaded_pdfs:
    st.write(f"Processing PDF: {pdf_doc.name}")
    form = st.form(key=f'question_form_{pdf_doc.name}')
    input_question = form.text_input("Type in your question", key=f'input_question_{pdf_doc.name}')
    submit_query = form.form_submit_button("Submit")

    if submit_query:
        start_time = time.time()  
        output = qa_instance.run(input_question)
        end_time = time.time()  
        processing_time = end_time - start_time 
        st.write("Output:")
        st.write(output)
        st.write(f"Time taken to process the query: {processing_time :.2f} seconds")
