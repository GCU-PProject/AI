import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin  # 상대경로를 절대경로로 합칠 때 쓰는 도구

# 설정 정보
BASE_DOMAIN = "https://leginfo.legislature.ca.gov"
INDEX_URL = "https://leginfo.legislature.ca.gov/faces/codes.xhtml"
HEADERS = {"User-Agent": "Mozilla/5.0"}
OUTPUT_FILE = "crawling_result.txt"  # 결과가 저장될 파일 이름

# ============== 1단계: 법령 목록 수집 ==============


def fetch_law_list():
    """
    [1단계] 메인 페이지에서 29개 법령의 [약어, 이름, 다음 단계 URL]을 추출합니다.
    """
    print(f"📡 법령 목록 페이지 접속 중... ({INDEX_URL})")

    try:
        res = requests.get(INDEX_URL, headers=HEADERS)
        soup = BeautifulSoup(res.text, "html.parser")

        law_list = []

        # 화면에 보이는 모든 링크(a 태그)를 하나씩 검사
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)  # 링크에 적힌 글자 (예: Vehicle Code - VEH)
            href = a["href"]  # 링크 주소 (예: /faces/codesTOCSelected.xhtml?...)

            # [필터링 로직]
            # 1. 주소에 'tocCode=' 가 있어야 함 (법령 링크의 특징)
            # 2. 텍스트에 ' - ' 가 있어야 함 (이름 - 약어 형식)
            if "tocCode=" in href and " - " in text:

                # 텍스트 분리: "Vehicle Code - VEH"  ->  ["Vehicle Code", "VEH"]
                # rsplit(" - ", 1): 오른쪽 끝에서부터 ' - '를 기준으로 딱 한 번만 자름
                title_part, abbr_part = text.rsplit(" - ", 1)

                law_info = {
                    "abbr": abbr_part.strip(),  # 약어 (예: VEH)
                    "title": title_part.strip(),  # 법령 이름 (예: Vehicle Code)
                    "next_url": urljoin(BASE_DOMAIN, href),  # 2단계로 갈 절대 주소 완성
                }

                law_list.append(law_info)

        print(f"✅ 총 {len(law_list)}개의 법령을 발견했습니다.")
        return law_list

    except Exception as e:
        print(f"🚨 1단계 에러 발생: {e}")
        return []


# ============== 2단계: 재귀적 심층 탐색 ==============
def fetch_deep_links(url, depth=1, visited_urls=None):
    """
    [2단계 개선판] 재귀(Recursion)를 이용해
    폴더(expandedbranch)면 계속 파고들고, 파일(displayText)이면 수집 목록에 담습니다.
    * 무한루프 방지(visited_urls) 및 Javascript 링크 무시 로직 적용
    """
    if visited_urls is None:
        visited_urls = set()

    # 안전장치: 너무 깊으면 중단
    if depth > 15:
        return []

    # 이미 방문한 곳(부모 폴더, 자기 자신)이면 중단 (Up 버튼 방어)
    if url in visited_urls:
        return []

    visited_urls.add(url)  # 방문 도장 쾅!

    # 로그 출력 (너무 길면 주석 처리 가능)
    # print(f"   {'  ' * depth}📡 탐색(Lv.{depth}): ...{url[-40:]}")

    try:
        res = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(res.text, "html.parser")

        collected_content_pages = []

        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            full_url = urljoin(BASE_DOMAIN, href)

            # 자기 자신을 가리키는 링크는 무시 (제자리 걸음 방지)
            if full_url == url:
                continue

            # [조건 A] displayText -> "여기가 본문이 있는 곳이다!" (수집 대상)
            # 재귀를 멈추고, 이 주소를 수집 목록에 넣습니다.
            if "displayText" in href:
                collected_content_pages.append(
                    {"category": text, "content_url": full_url}
                )

            # [조건 B] displayexpandedbranch -> "여기는 폴더다!" (재귀 대상)
            # 더 깊이 들어갑니다. (단, 이미 방문한 URL이면 위에서 걸러짐)
            elif "displayexpandedbranch" in href:
                sub_results = fetch_deep_links(full_url, depth + 1, visited_urls)
                collected_content_pages.extend(sub_results)

                time.sleep(0.1)  # 서버 부하 방지용 미세 딜레이

        return collected_content_pages

    except Exception as e:
        print(f"      🚨 2단계 탐색 에러: {e}")
        return []


# ============== 3단계 ==============


def parse_article_content(url):
    """
    [3단계] 페이지 내의 모든 조항(Section)을 찾아서
    각각 분리된 딕셔너리 리스트로 반환합니다.
    """
    try:
        res = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(res.text, "html.parser")

        # 1. 페이지에 있는 '모든' 조항 제목을 다 찾습니다. (리스트 형태)
        h6_tags = soup.find_all("h6")

        if not h6_tags:
            return []  # 조항이 하나도 없으면 빈 리스트 반환

        collected_sections = []  # 수집된 조항들을 담을 바구니

        # 2. 각 조항 제목(h6)마다 반복하면서 본문을 챙깁니다.
        for h6 in h6_tags:

            # (1) 제목 검사 (SEC/SECTION 시작 확인)
            title_text = h6.get_text(strip=True).upper()

            # 첫 글자가 존재하는지 확인 후, 숫자인지 검사 (title_text[0].isdigit())
            is_numeric_start = len(title_text) > 0 and title_text[0].isdigit()

            # Preamble 등은 건너뛰기
            if not (
                title_text.startswith("SEC")
                or title_text.startswith("SECTION")
                or is_numeric_start
            ):
                print(f"         ⚠️ [Skip] 조항 아님: {title_text}")
                continue  # 이번 h6는 버리고 다음 h6로 넘어감

            # (2) 조항 번호 정제 (삭제했음)
            clean_article_no = title_text

            # (3) 본문 챙기기 (★핵심 로직 변경★)
            # "내 바로 밑 동생들(siblings)을 훑어보다가,
            # 다음 h6(다른 조항 제목)가 나오면 즉시 멈춘다!"
            content_lines = []

            for sibling in h6.next_siblings:
                # 만약 동생이 또 다른 제목(h6)이라면? -> 내 구역 끝! 멈춰!
                if sibling.name == "h6":
                    break

                # p 태그인 경우에만 내용 담기
                if sibling.name == "p":
                    text = sibling.get_text(strip=True)
                    style = sibling.get("style", "")

                    if text and "font-size:0.9em" not in style:
                        content_lines.append(text)

            # (4) 하나의 조항 완성 -> 바구니에 담기
            section_data = {
                "article_no": clean_article_no,  # 예: SEC 1
                "content": "\n".join(content_lines),  # SEC 1에 속한 문단들만 합침
            }
            collected_sections.append(section_data)

        return collected_sections  # [ {SEC1}, {SEC2}, ... ] 리스트 반환

    except Exception as e:
        print(f"         🚨 3단계 파싱 에러: {e}")
        return []


# ============== 메인 실행부 ==============

if __name__ == "__main__":

    # 함수 실행
    all_laws = fetch_law_list()

    # 결과 확인
    print("\n🔎 수집된 법령 목록 :")
    for i, law in enumerate(all_laws):
        print(f"[{i+1}/{len(all_laws)}] {law['title']} ({law['abbr']})")
        print(f"    🔗 이동할 주소: {law['next_url']}")
    print("=" * 60)

    # 타겟 법령 코드
    target_code = "GOV"

    target_laws = [law for law in all_laws if law["abbr"] == target_code]

    print(f"\n🎯 집중 공략할 법령: {target_code}")
    print(
        f"📂 저장될 파일명: crawling_map_{target_code}.json / law_data_{target_code}.jsonl"
    )

    # 2. 모든 본문 페이지(displayText) 찾기 (재귀 탐색 시작)
    print("   📡 전체 구조 탐색 중... (잠시만 기다려주세요)")

    raw_content_pages = []

    for i, law in enumerate(target_laws):
        # 여기서 2단계 함수가 바닥 끝까지 훑어서 [본문URL 리스트]를 가져옵니다.
        law_pages = fetch_deep_links(law["next_url"])
        # 법령 코드 추가. 나중에 출력할 때 쓰기 위함입니다.
        for page in law_pages:
            page["law_code"] = law["abbr"]  # 예: 'BPC' 저장
        raw_content_pages.extend(law_pages)
    print(
        f"\n   ✅ 총 {len(raw_content_pages)}개의 본문 페이지를 찾았습니다. 데이터 수집 시작...\n"
    )

    # ========================================================
    # ★ [수정 2] 중복 제거 및 고유번호(map_id) 부여 로직 추가
    # ========================================================
    all_content_pages = []
    seen_urls = set()

    print("   🧹 중복 제거 및 ID 부여 중...")

    for item in raw_content_pages:
        # URL이 이미 등록된 적 있다면 건너뜀 (중복 방지)
        if item["content_url"] in seen_urls:
            continue

        seen_urls.add(item["content_url"])

        # map_id 부여 (1부터 시작)
        item["map_id"] = len(all_content_pages) + 1
        all_content_pages.append(item)

    print(
        f"   ✅ 최종 정리 완료: 총 {len(all_content_pages)}개 (중복 {len(raw_content_pages) - len(all_content_pages)}개 제거됨)\n"
    )
    # ========================================================
    # 지도(URL리스트)를 파일로 저장하기
    # ========================================================
    map_filename = f"crawling_map_{target_code}.json"
    with open(map_filename, "w", encoding="utf-8") as f:
        json.dump(all_content_pages, f, ensure_ascii=False, indent=2)

    print(f"   💾 [안전장치] 지도가 '{map_filename}' 파일로 저장되었습니다.")
    print(f"   (혹시 멈추면 이 파일을 열어서 확인하거나 로드할 수 있습니다.)\n")

    # ========================================================
    # 데이터 수집 및 저장
    # ========================================================
    data_filename = f"law_data_{target_code}.jsonl"
    with open(data_filename, "w", encoding="utf-8") as f:
        # 찾은 본문 페이지들을 하나씩 방문해서 텍스트 추출
        for j, item in enumerate(all_content_pages):
            law_code = item.get("law_code", "Unknown")
            map_id = item.get("map_id")
            print(
                f"   [{j+1}/{len(all_content_pages)}] (ID: {map_id}) Parsing: {item['category']}"
            )
            print(f"       📌 법령: {law_code} | 🔗 URL: {item['content_url']}")

            # 3단계 함수 호출
            sections_list = parse_article_content(item["content_url"])

            if sections_list:
                for sec in sections_list:
                    # (1) 저장할 데이터 딕셔너리 생성
                    row = {
                        "map_id": map_id,
                        "law_code": law_code,
                        "category": item["category"],
                        "article_no": sec["article_no"],
                        "content": sec["content"],
                        "url": item["content_url"],
                    }

                    # (2) 파일에 쓰기 (JSON 변환 + 줄바꿈)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()  # 안전하게 즉시 저장

                    # (3) 화면 출력
                    print(
                        f"      💾 수집 성공: {sec['article_no']} (길이: {len(sec['content'])} - URL: {item['content_url']})"
                    )

            else:
                print(f"      ⚠️ 내용 없음 (URL: {item['content_url']})")

            time.sleep(0.5)  # 페이지 간 휴식

    print(f"\n🎉 전체 {len(all_content_pages)}개의 법령 수집 완료!")
