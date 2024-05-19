# DocuTalk

DocuTalk is a Streamlit application that allows you to upload various document files (PDF, DOCX, PPTX) and ask questions about the content of the documents. The application uses the LangChain library and the Ollama language model to process the documents, generate embeddings, and provide answers to the user's questions.

## Requirements

To run this application, you need to have the following dependencies installed:

```
pip install streamlit langchain_community pypdf2 chromadb langchain python-docx python-pptx ollama pull mistral
ollama pull nomic-embed-text
```

## Usage

1. Run the application using the following command:

```
streamlit run app.py
```

2. Upload your documents (PDF, DOCX, PPTX) by clicking the "Upload files" button.

3. Once the documents are uploaded, click the "Generate Embeddings" button to process the documents and create the necessary embeddings.

4. Enter your question in the text input field and click the "Query Documents" button to get the answer based on the uploaded documents.

5. If you have any CSV files uploaded, the application will automatically detect them and display the data in a plot.

## Features

- Supports PDF, DOCX, and PPTX file formats.
- Generates embeddings for the uploaded documents using the Ollama language model.
- Provides answers to user questions based on the content of the uploaded documents.
- Plots data from uploaded CSV files.

## Acknowledgements

This application uses the following libraries and tools:

- [Streamlit](https://streamlit.io/) for the web application framework.
- [LangChain](https://langchain.com/) for the language model and document processing.
- [PyPDF2](https://pypi.org/project/PyPDF2/) for PDF file processing.
- [python-docx](https://python-docx.readthedocs.io/) for DOCX file processing.
- [python-pptx](https://python-pptx.readthedocs.io/) for PPTX file processing.
- [Chroma](https://www.trychroma.com/) for the vector database.
- [Ollama](https://www.anthropic.com/models) for the language model.
