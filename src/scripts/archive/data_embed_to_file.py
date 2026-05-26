import asyncio
import json
import os
import sys

# ------------------------------------------------------------------------------
# 1. 모듈 경로 설정
# ------------------------------------------------------------------------------
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
import vertexai
from src.core.config import settings

# ------------------------------------------------------------------------------
# 2. 전역 설정
# ------------------------------------------------------------------------------
if settings.GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        settings.GOOGLE_APPLICATION_CREDENTIALS
    )


async def step1_embed_to_file():
    # --------------------------------------------------
    # 3. 초기화 (Vertex AI Only) - DB 연결 제거됨
    # --------------------------------------------------
    print(f"🔧 설정 로드 완료: Project={settings.GCP_PROJECT_ID}")
    print("🔄 GCP Vertex AI 초기화 중...")
    vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    # --------------------------------------------------
    # 4. 파일 탐색
    # --------------------------------------------------
    # files = glob.glob("law_data_VEH.jsonl")
    # 원상복귀: 위 주석을 풀고 아래 리스트를 주석 처리하세요.
    files = [
        "data/law_data_AU.jsonl",
    ]

    # 파일 존재 여부 확인 (리스트 사용 시 glob과 달리 직접 확인 필요)
    files = [f for f in files if os.path.exists(f)]

    if not files:
        print("🚨 대상 파일이 없습니다.")
        return

    print(f"📂 발견된 데이터 파일: {files}")

    # --------------------------------------------------
    # 5. 데이터 처리 루프 (DB 세션 제거됨)
    # --------------------------------------------------
    for filename in files:
        output_filename = filename.replace(".jsonl", "_embedded.jsonl")  # 결과 파일명
        print(f"📏 개수 세는 중: {filename}...", end="\r")

        with open(filename, "r", encoding="utf-8") as f_cnt:
            total_lines = sum(1 for line in f_cnt if line.strip())

        print(f"\n🚀 파일 처리 시작: {filename} -> {output_filename}")

        # 입력 파일(f) 읽기 & 출력 파일(f_out) 쓰기 모드로 열기
        with (
            open(filename, "r", encoding="utf-8") as f,
            open(output_filename, "w", encoding="utf-8") as f_out,
        ):

            batch_data = []  # 임베딩용 텍스트 리스트
            batch_objects = []  # JSON 객체 리스트 (DB 객체 아님)
            current_count = 0

            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)

                # (A) 임베딩 텍스트 생성 (로직 유지)
                text_for_embedding = f"{row.get('law_type', '')} {row.get('article_no', '')} {row.get('content', '')}"
                batch_data.append(text_for_embedding)

                # (B) 객체 보관 (DB 객체 생성 대신 딕셔너리 그대로 사용)
                # 나중에 DB에 넣을 때 매핑할 정보들을 그대로 가져갑니다.
                batch_objects.append(row)

                # (C) 배치 처리
                BATCH_SIZE = 10
                if len(batch_data) >= BATCH_SIZE:
                    current_count += len(batch_objects)
                    await process_batch_to_file(
                        embedding_model,
                        batch_data,
                        batch_objects,
                        f_out,  # 파일 핸들 전달
                        current_count,
                        total_lines,
                    )
                    batch_data = []
                    batch_objects = []

            # (D) 남은 데이터 처리
            if batch_data:
                current_count += len(batch_objects)
                await process_batch_to_file(
                    embedding_model,
                    batch_data,
                    batch_objects,
                    f_out,
                    current_count,
                    total_lines,
                )

        print(f"✅ 파일 변환 완료: {output_filename} (총 {current_count}건)")

    print("\n🎉 1단계 완료! 이제 2단계 스크립트를 실행하세요.")


async def process_batch_to_file(model, texts, objects, file_handle, current, total):
    """
    텍스트 배치를 받아 임베딩을 생성하고 '파일'에 저장하는 함수
    """
    try:
        # 1. Vertex AI 호출 (로직 유지)
        inputs = [
            TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT") for t in texts
        ]
        embeddings = model.get_embeddings(inputs)

        # 2. 결과 매핑 및 파일 쓰기 (DB 저장 -> 파일 쓰기로 변경)
        for obj, emb in zip(objects, embeddings):
            # 원본 객체에 'embedding' 필드 추가
            obj["embedding"] = emb.values

            # JSON 라인으로 저장
            file_handle.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # 파일은 버퍼링될 수 있으므로 강제 저장(flush)
        file_handle.flush()

        # 3. 진행률 표시 및 대기 (로직 유지)
        progress = (current / total) * 100
        print(
            f"   💾 파일 저장: {current}/{total}건 ({progress:.1f}%) -> 13초 대기... 💤"
        )

        await asyncio.sleep(13)

    except Exception as e:
        print(f"\n   🚨 배치 처리 실패: {e}")
        # 파일 처리는 롤백이 없으므로 에러 로그만 출력


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(step1_embed_to_file())
