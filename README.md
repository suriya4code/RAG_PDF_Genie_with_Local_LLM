# PDF Genie

##### input file used for this ml model => *"IRS tax filling instructions 2024.pdf" (100+ pages)*

You can ask any questions to this local ML model and it will respond based on information/context fed through pdf file.

##### Sample output using Streamlit :

<img src="https://raw.githubusercontent.com/suriya4code/RAG_PDF_Genie_with_Local_LLM/main/output/04_streamlit_q1.png" alt="grag system st">

##### Running Locally: 

<img src="https://raw.githubusercontent.com/suriya4code/RAG_PDF_Genie_with_Local_LLM/main/output/01_question1.png" alt="rag system q1">


<img src="https://raw.githubusercontent.com/suriya4code/RAG_PDF_Genie_with_Local_LLM/main/output/02_question2.png" alt="rag system q2">

##### Spinning up Streamlit:

<img src="https://raw.githubusercontent.com/suriya4code/RAG_PDF_Genie_with_Local_LLM/main/output/03_streamlit_running.png" alt="rag system q2">

##### Question 1 through Streamlit local UI: 

<img src="https://raw.githubusercontent.com/suriya4code/RAG_PDF_Genie_with_Local_LLM/main/output/04_streamlit_q1.png" alt="grag system st">

##### Question 2 through Streamlit local UI: 

<img src="https://raw.githubusercontent.com/suriya4code/RAG_PDF_Genie_with_Local_LLM/main/output/05_streamlit_q2.png" alt="grag system st">


# Overview

PDF Genie is a PDF search engine pipeline that processes PDF documents, extracts text, and performs various operations using the LangChain library and other tools. It is designed to build a Retrieval-Augmented Generation (RAG) system, which combines information retrieval and natural language generation to provide accurate and contextually relevant answers from PDF documents.


## Features

- Load and process PDF documents
- Split text using recursive character text splitter
- Create vector stores for efficient text retrieval
- Use embeddings for text representation
- Perform multi-query retrieval
- Generate responses using a language model

## Requirements

- Python 3.7+
- `langchain_community`
- `langchain_text_splitters`
- `langchain_community.vectorstores`
- `langchain_ollama`
- `langchain.prompts`
- `langchain_core`
- `ollama`
- `urllib3`


