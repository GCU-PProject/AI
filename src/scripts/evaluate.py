# src/scripts/evaluate.py
"""
[참고 코드] 테디노트 CH16 - RAGAS 평가 스크립트

이 코드는 테디노트님의 RAG 비법노트를 참고한 것으로,
우리 프로젝트(evaluate_rag.py)를 작성할 때 참고용으로 사용합니다.

[전체 흐름]
1. 합성 데이터셋(CSV) 로드
2. RAG 파이프라인(PDF→FAISS→Retriever→LLM) 구성
3. 테스트셋의 모든 질문에 대해 RAG 답변 생성
4. RAGAS로 4개 지표(정밀도, 충실도, 관련성, 재현율) 평가
"""

# =========================================================
# 1. 환경 설정
# =========================================================
# .env 파일에서 API 키를 불러옵니다 (OpenAI API Key 등)
from dotenv import load_dotenv

load_dotenv()

# LangSmith: LangChain 체인 실행 과정을 대시보드로 시각화하는 도구
# langchain-teddynote는 테디노트 전용 패키지 → 우리 프로젝트에서는 제외
from langchain_teddynote import logging

logging.langsmith("CH16-Evaluations")

# =========================================================
# 2. 합성 데이터셋 로드
# =========================================================
# generate.py에서 만든 CSV 파일을 불러옵니다.
# 이 CSV에는 question, contexts, ground_truths, answer 컬럼이 있습니다.
import pandas as pd

df = pd.read_csv("data/ragas_synthetic_dataset.csv")
df.head()  # Jupyter에서 상위 5개 행 미리보기 (스크립트에서는 효과 없음)

# =========================================================
# 3. Pandas DataFrame → HuggingFace Dataset 변환
# =========================================================
# RAGAS는 HuggingFace의 Dataset 형식을 입력으로 받습니다.
# Pandas DataFrame을 Dataset으로 변환합니다.
from datasets import Dataset

test_dataset = Dataset.from_pandas(df)
test_dataset  # 데이터셋 정보 출력


# =========================================================
# 4. contexts 컬럼 변환 (문자열 → 리스트)
# =========================================================
# CSV에서 리스트는 문자열로 저장됩니다.
# "['문서1', '문서2']" (str) → ['문서1', '문서2'] (list)로 변환합니다.
#
# ast.literal_eval(): 문자열을 안전하게 Python 리터럴로 변환하는 함수
# .map(): 데이터셋의 모든 행에 함수를 적용 (각 행이 example으로 전달됨)
def convert_to_list(example):
    contexts = ast.literal_eval(example["contexts"])
    return {"contexts": contexts}


test_dataset = test_dataset.map(convert_to_list)
print(test_dataset)
test_dataset[1]["contexts"]  # 변환 결과 확인

# =========================================================
# 5. RAG 파이프라인 구성
# =========================================================
# 테디노트 코드: PDF → 청킹 → FAISS → Retriever → LLM
# 우리 코드에서는: DB(pgvector) → 커스텀 Retriever → LLM (이미 존재)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 단계 1: PDF 문서 로드
# → 우리는 DB에서 법률 데이터를 가져오므로 이 단계 불필요
loader = PyMuPDFLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")
docs = loader.load()

# 단계 2: 문서 분할 (1000자 단위로 청킹, 50자 겹침)
# → 우리는 법률 조항 단위로 이미 분할되어 있으므로 불필요
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

# 단계 3: 임베딩 모델 생성
# → 우리는 VertexAIEmbeddings 사용
embeddings = OpenAIEmbeddings()

# 단계 4: 벡터스토어 생성 (FAISS = 메모리 기반 벡터 DB)
# → 우리는 PostgreSQL + pgvector 사용
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 단계 5: 검색기 생성
# .as_retriever(): 벡터스토어에서 검색 기능을 가진 객체 생성
# → 우리는 커스텀 retrieve_laws() 함수 사용
retriever = vectorstore.as_retriever()

# 단계 6: 프롬프트 생성
# {context}: 검색된 문서가 들어갈 자리
# {question}: 사용자 질문이 들어갈 자리
# → 우리는 chat.yaml의 CHAT_PROMPT 사용
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 

#Context: 
{context}

#Question:
{question}

#Answer:"""
)

# 단계 7: LLM 생성
# → 우리는 ChatVertexAI(Gemini) 사용
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 단계 8: LCEL 체인 생성 ⭐ (RunnablePassthrough 패턴)
# {"context": retriever, "question": RunnablePassthrough()}
#   → retriever: 질문을 받으면 자동으로 검색 실행 → 결과를 context에 넣음
#   → RunnablePassthrough(): 질문을 그대로 통과시켜 question에 넣음
# | prompt: context와 question을 프롬프트에 삽입
# | llm: 프롬프트를 LLM에 전달하여 답변 생성
# | StrOutputParser(): LLM 응답에서 텍스트만 추출
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# =========================================================
# 6. 테스트셋 질문에 대한 RAG 답변 일괄 생성
# =========================================================
# 테스트셋에서 질문만 리스트로 추출
batch_dataset = [question for question in test_dataset["question"]]
batch_dataset[:3]  # 미리보기

# .batch(): 여러 질문을 한꺼번에 처리 (병렬 실행)
# .invoke()는 1개씩, .batch()는 리스트 전체를 처리
answer = chain.batch(batch_dataset)
answer[:3]  # 답변 미리보기

# 'answer' 컬럼을 데이터셋에 추가 (이미 있으면 덮어쓰기)
# HuggingFace Dataset은 컬럼을 직접 수정할 수 없어서,
# 기존 컬럼을 삭제 후 새로 추가하는 방식으로 덮어씁니다.
if "answer" in test_dataset.column_names:
    test_dataset = test_dataset.remove_columns(["answer"]).add_column("answer", answer)
else:
    test_dataset = test_dataset.add_column("answer", answer)

# =========================================================
# 7. RAGAS 평가 실행
# =========================================================
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,   # 답변 관련성: 답변이 질문에 맞는 내용인가?
    faithfulness,       # 충실도: 답변이 검색된 문서에만 근거했는가? (할루시네이션 체크)
    context_recall,     # 검색 재현율: 정답에 필요한 정보를 모두 검색했는가?
    context_precision,  # 검색 정밀도: 검색된 문서가 정답에 관련 있는가?
)

# evaluate(): 데이터셋과 지표를 전달하면 RAGAS가 LLM으로 자동 채점
# 내부적으로 각 지표마다 LLM을 여러 번 호출하여 채점합니다.
# (질문 1개당 약 5~10회 LLM 호출 × 질문 수 = 총 호출 수)
result = evaluate(
    dataset=test_dataset,
    metrics=[
        context_precision,   # 검색 정밀도
        faithfulness,        # 충실도
        answer_relevancy,    # 답변 관련성
        context_recall,      # 검색 재현율
    ],
)

# =========================================================
# 8. 결과 확인
# =========================================================
# 전체 평균 점수 출력 (0~1 사이, 1에 가까울수록 좋음)
# 예: {'context_precision': 0.85, 'faithfulness': 0.90, ...}
result

# 각 질문별 상세 점수를 DataFrame으로 변환
result_df = result.to_pandas()
result_df.head()  # 상위 5개 질문의 점수 확인

# 특정 지표 컬럼만 추출하여 확인
result_df.loc[:, "context_precision":"context_recall"]
