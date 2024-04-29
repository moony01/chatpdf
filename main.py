from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import tempfile
import os

#Open api key load
load_dotenv()

# 제목
st.title("ChatPDF")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("Choose a file")
st.write("---")

def pdf_to_document(uploaded_files):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 파일 업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=300,             # 몇글자 단위로 쪼갤지
        chunk_overlap=20,           # 문백 유지 글자 단위
        length_function=len,        # len은 길이를 구하는 함수
        is_separator_regex=False,   # 글자를 정규 표현식으로 자를지
    )

    texts = text_splitter.split_documents(pages)

    # load it into Chroma
    vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())

    # Question
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input('질문을 입력하세요.')

    if st.button('질문하기'):
        # Prompt
        template = '''다음 context를 토대로 질문에 답하고 최소 10글자 이상 30글자 이하의 문장으로 대답해줘:
        {context}

        Question: {question}
        '''

        prompt = ChatPromptTemplate.from_template(template)

        # LLM
        model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=40)

        # Rretriever(검색)
        # 사용자의 질문이나 주어진 컨텍스트에 가장 관련된 정보를 찾아내는 과정입니다. 
        # 사용자의 입력을 바탕으로 쿼리를 생성하고, 인덱싱된 데이터에서 가장 관련성 높은 정보를 검색합니다. 
        # LangChain의 retriever 메소드를 사용합니다.
        retriever = vectorstore.as_retriever()

        # Combine Documents
        def format_docs(docs):
            return '\n\n'.join(doc.page_content for doc in docs)

        # RAG Chain 연결
        rag_chain = (
            {'context': retriever | format_docs, 'question': RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        # Chain 실행
        result = rag_chain.invoke(question)
        print(result)
        st.write(result)