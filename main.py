from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#Open api key load
load_dotenv()

#Loader
loader = PyPDFLoader("unsu.pdf")
pages = loader.load_and_split()

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
question = "아내가 먹고 싶어하는 음식은 무엇이야?"
docs = vectorstore.similarity_search(question)
# print(len(docs))
# print(docs)
# print(docs[0].page_content)

# Prompt
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0)

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