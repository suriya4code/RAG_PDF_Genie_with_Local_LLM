"""
Steps to Build a PDF Search Engine with PDF Genie:

1. PDF Input Handling
 Handle incoming PDF files for processing.

2. Text Segmentation
 Divide extracted text into smaller segments for efficient processing.

3. Model Embedding
 Generate vector representations of the segmented text using an embedding model.

4. Vector Storage
 Store the generated embeddings in a vector database.

5. Similarity Search
 Perform a similarity search on the vector database to identify top matches.

6. Result Delivery
Return the most similar documents from the search results for user review or analysis.
"""

import os
import time
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama



import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = logging.Logger("PDF Genie")
# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter('[%(asctime)s] %(levelname)-5s %(message)s')
console.setFormatter(formatter)
log.addHandler(console)
log.info("PDF Genie is starting...")

# Constants
# DOC_PATH = "./data/fact_sheet.pdf"
DOC_PATH = "./data/IRS_instruction_2024.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"

def calc_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        log.info(f"Execution time of {func.__name__}: {execution_time:.4f} seconds")
        return result
    return wrapper

# Step 1: PDF Input Handling

@calc_time
def load_pdf_document(doc_path):
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        log.info("Step 1: PDF loaded successfully.")
        return data
    else:
        log.error(f"PDF file not found at path: {doc_path}")
        return None


# Step 2: Text Segmentation
@calc_time
def segment_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    segments = splitter.split_documents(text)
    log.info(f"Step 2: Text segmented into {len(segments)} segments.")
    return segments


# Step 3& 4 : Model Embedding and Vector Generation
@calc_time
def embed_text_and_create_vector_db(chunks):
    # pull the model if not already available
    ollama.pull(EMBEDDING_MODEL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
    )
    log.info("Step 3 : Vector embeddings generated and stored.")
    return vector_db


# Step 5: Similarity Search

## create retriever object
@calc_time
def create_retriever(vector_db, llm):
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model which will inspect a pdf file and 
            help user with questions on that file. Your task is to generate three
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question.
            Provide these alternative questions separated by newlines.
            Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    log.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain"""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
                {context}
                Question: {question}
                """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    log.info("Chain created successfully.")
    return chain


def main():
    """Main function to orchestrate the PDF search engine pipeline."""
    # Step 1: Load PDF Document
    pdf_data = load_pdf_document(DOC_PATH)

    if pdf_data is None:
        return 
    
    # Step 2: Segment Text
    chunks = segment_text(pdf_data)

    # Step 3 & 4: Embed Text and Generate Vectors
    vector_db = embed_text_and_create_vector_db(chunks)

    # Step 5: Create Retriever
    llm = ChatOllama(model=MODEL_NAME)
    retriever = create_retriever(vector_db, llm)

    # Step 6: Create Chain
    chain = create_chain(retriever, llm)

    sample_question = "What are the tax filling steps for married couple?"

    # Run the chain
    res = chain.invoke(input = sample_question)
    log.info(f"Response: {res}")


if __name__ == "__main__":
    main()