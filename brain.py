import re
from io import BytesIO
from typing import Tuple, List
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
import faiss
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import os

class TicketResolution(BaseModel):
    class QuizQuestion(BaseModel):
        question: str = Field(description="question on the required topic.")
        options: list[str] = Field(description="randomized list of options")
        answer : str  = Field(description="Answer of the question")
        explanation : str = Field(description="very short explanations why the option is right or wrong under 25 words")

    questions: list[QuizQuestion]
    
    confidence: float = Field(description="Confidence in the resolution (0-1)")



def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output, filename

def text_to_docs(text: List[str], filename: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc.metadata["filename"] = filename
            doc_chunks.append(doc)
    return doc_chunks

def docs_to_index(docs, openai_api_key):
    index = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))
    return index

def get_index_for_pdf(pdf_files, pdf_names, openai_api_key):
    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents = documents + text_to_docs(text, filename)
    index = docs_to_index(documents, openai_api_key)
    return index

def generate_quiz_questions(pdf_extract: str, client: OpenAI, model: str):
    system_prompt = system_prompt = """
You are a AI Quiz Bot. You will be provided with a question,
and 4 options, including the answer for the questions.Never provide the first option as answer,
And you will generate question based on difficulty level 1 to 20
"""
 

    query = f"""
    Generate 10 questions based on the following PDF extract:
    {pdf_extract}
    
    never mention about the question source or pdf extract in the question.
    also dont ask questions about the page number,include formulas if present in the file
    be as realistic as possible.
    """

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        response_format=TicketResolution,
    )

    response_pydantic =  completion.choices[0].message.parsed
    response = response_pydantic.model_dump()
    
    print(type(response))
    print(response)
    return response

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

# Set up the OpenAI client using the API key from the environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    generate_quiz_questions(pdf_extract="hey i am tamim" , model="gpt-4o-2024-08-06",client=client)
