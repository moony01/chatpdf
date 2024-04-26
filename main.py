from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

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

#Embedding
embeddings_model = OpenAIEmbeddings()

# load it into Chroma
db = Chroma.from_documents(texts, embeddings_model)

# Question
question = "아내가 먹고 싶어하는 음식은 무엇이야?"
llm = ChatOpenAI(
        temperature=0,
        max_tokens=1000
      )
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), llm=llm
)

docs = retriever_from_llm.get_relevant_documents(query=question)
print(len(docs))
print(docs)