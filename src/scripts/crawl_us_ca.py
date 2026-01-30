import sys
import os
import time
import requests
import psycopg2
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

# 경로 설정
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.core.config import settings


# 1. DB 연결
def get_db_connection():
    return psycopg2.connect(**settings.CRAWLER_DB_PARAMS)


# [설정] URL 정보
BASE_URL = "https://leginfo.legislature.ca.gov"
# (1) 법률 이름표를 배우러 갈 곳 (목록 페이지)
INDEX_URL = "https://leginfo.legislature.ca.gov/faces/codes.xhtml"
# (2) 실제 데이터를 캐러 갈 곳 (차량법 Division 11)
TARGET_URL = "https://leginfo.legislature.ca.gov/faces/codes_display_expandedbranch.xhtml?tocCode=VEH&division=11.&title=&part=&chapter=&article="

HEADERS = {"User-Agent": "Mozilla/5.0"}


# [신규 기능] 웹사이트에서 법률 이름 명단(Map)을 직접 만들어오는 함수
def build_dynamic_code_map():
    print(f"📖 법률 이름 목록을 학습하러 갑니다... ({INDEX_URL})")
    try:
        res = requests.get(INDEX_URL, headers=HEADERS)
        soup = BeautifulSoup(res.text, "html.parser")

        dynamic_map = {}

        # HTML에서 링크를 찾아서 약어와 풀네임을 분리
        # 예: <a ...> Vehicle Code - VEH </a>
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)

            # " - "가 포함된 링크만 법률 링크로 간주
            if " - " in text and "tocCode=" in a["href"]:
                # 오른쪽 끝에서 한 번만 쪼갬 ("California - Law - ABC" 같은 경우 대비)
                title, abbr = text.rsplit(" - ", 1)
                dynamic_map[abbr] = title.strip()

        print(f"✅ 학습 완료! 총 {len(dynamic_map)}개의 법률 이름을 익혔습니다.")
        return dynamic_map

    except Exception as e:
        print(f"⚠️ 목록 학습 실패 (기본값 사용): {e}")
        return {}  # 실패하면 빈 딕셔너리 반환


def get_law_title(url, code_map):
    """URL에서 약어를 뽑고, 학습한 code_map에서 풀네임을 찾음"""
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        # URL에서 VEH 같은 약어 추출
        code_abbr = params.get("tocCode", params.get("lawCode", [""]))[0]

        # 학습한 맵에서 찾기 (없으면 그냥 약어 반환)
        return code_map.get(code_abbr, f"California Law ({code_abbr})")
    except:
        return "Unknown Law"


def parse_section_page(html):
    """상세 페이지 파싱 (기존과 동일)"""
    soup = BeautifulSoup(html, "html.parser")
    h6_tag = soup.find("h6")
    if not h6_tag:
        return None

    raw_num = h6_tag.get_text().strip().replace(".", "")
    article_no = f"VEH {raw_num}"

    content_lines = []
    for sibling in h6_tag.next_siblings:
        if sibling.name == "p":
            text = sibling.get_text().strip()
            style = sibling.get("style", "")
            if (
                text
                and "font-size:0.9em" not in style
                and not text.startswith("(Enacted")
            ):
                content_lines.append(text)

    return {"article_no": article_no, "content": "\n".join(content_lines)}


def main():
    print(f"🚀 크롤러 가동 시작 (Target DB: {settings.DB_HOST})...")

    # [1단계] 먼저 법률 이름들을 배워옵니다. (동적 매핑 생성)
    code_map = build_dynamic_code_map()

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # [2단계] 실제 크롤링 시작
        print(f"📡 상세 목록 수집 중: {TARGET_URL}")
        res = requests.get(TARGET_URL, headers=HEADERS)
        soup = BeautifulSoup(res.text, "html.parser")

        links = []
        for a in soup.find_all("a", href=True):
            if "displaySection" in a["href"]:
                links.append(BASE_URL + a["href"])

        links = sorted(list(set(links)))
        print(f"✅ 총 {len(links)}개의 조항 발견.")

        # [3단계] 데이터 저장
        for i, link in enumerate(links[:5]):  # 테스트용 [:5]
            print(f"[{i+1}/{len(links)}] 처리 중... ", end="")

            sub_res = requests.get(link, headers=HEADERS)
            data = parse_section_page(sub_res.text)

            if data:
                # 아까 만든 code_map을 이용해 이름을 가져옵니다.
                dynamic_title = get_law_title(link, code_map)

                # 카테고리는 여전히 고정 (이 스크립트는 Division 11 전용이므로)
                fixed_category = "Division 11"

                sql = """
                INSERT INTO laws (country_id, jurisdiction, law_title, category, article_no, content, source_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(
                    sql,
                    (
                        1,
                        "US-CA",
                        dynamic_title,  # ✅ 이제 웹사이트에서 긁어온 진짜 이름을 씁니다!
                        fixed_category,
                        data["article_no"],
                        data["content"],
                        link,
                    ),
                )
                conn.commit()
                print(f"✅ 저장 ({data['article_no']}) -> 법률명: {dynamic_title}")
            else:
                print("❌ 파싱 실패")

            time.sleep(0.5)

    except Exception as e:
        print(f"\n🚨 에러 발생: {e}")
    finally:
        if conn:
            conn.close()
        print("👋 작업 종료")


if __name__ == "__main__":
    main()
