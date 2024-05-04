## 프로젝트 제목
PDF 파일 업로드 AI 채팅: [ai-poet-moony.streamlit.app](https://chatpdf-moony.streamlit.app/)

## 프로젝트 설명
Langchain framework를 활용한 PDF 파일을 업로드 하여 AI에게 학습 시킨 뒤 PDF파일의 데이터를 기반으로 대화를 하는 앱

## 사용기술
* Langchain
* ChromaDB
* Retriever
* Streamlit
* Streamlit Extras
* Python
* RAGs

## 설치 방법
1. 앱은 Python기반입니다. 앱을 실행하기 위해 [Python v3.9을 설치](https://www.python.org/downloads/) 합니다. (`v3.9`를 기반으로 개발하였습니다.)

2. 저장소를 clone합니다.

3. 앱에 import하고있는 library를 설치해줍니다.

```bash
pip install langchain
pip install openai
pip install pypdf
pip install chromadb
pip install tiktoken
pip install pysqlite3-binary
pip install langchain-openai
pip install langchain-chroma
pip install streamlit-extras
```

4. main.py를 실행할 경우 코드 맨 위 `로컬로 테스트 할 땐 아래 코드 주석을 해제해야함` 아래 코드를 주석 해제합니다.

5. main.py를 실행할 경우 코드 맨 위 `로컬로 테스트 시 주석 처리 해야함` 아래 코드를 주석 처리 해줍니다.

6. 라이브러리 설치를 완료했으면 OpenAI는 `streamlit run main.py` (Key 관련 에러는 .env 파일을 생성 후 키를 발급 받아 입력해야합니다.)

```bash
streamlit run main.py
```