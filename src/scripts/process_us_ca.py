import json
import re
from datasets import load_dataset

# ================= 설정 =================
TARGET_PER_COUNTRY = 500
OUTPUT_FILE = "laws_data.json"

COUNTRY_MAP = {"US": 1, "CA": 2}

# ================= 로직 =================


def clean_text(text):
    if not text:
        return ""

    # 1. 시스템 헤더 및 변환기 로그 제거
    # "Online@...", "USCConverter" 패턴 삭제
    text = re.sub(r".*?Online@[\w\-]+\s+(yes|no)", "", text, flags=re.IGNORECASE)
    text = re.sub(r".*?USCConverter\s+[\d\.]+", "", text, flags=re.IGNORECASE)

    # 2. 불필요한 헤더/푸터 제거
    text = re.sub(r"Title\s+\d+\s+USC.*", "", text)
    text = re.sub(
        r"Current through.*", "", text
    )  # "Current through 117-49" 같은 버전 정보 삭제

    # 3. 목차 제거
    # "301.Title... 303.Title..."과 같이 숫자가 반복해서 나오는 패턴 제거
    if len(re.findall(r"\d{3,}\.[A-Z]", text)) > 3:
        return ""

    # 4. 개정 이력 자르기
    cutoff_markers = [
        "Editorial Notes",
        "Statutory Notes",
        "git Historical and Revision",
        "Amendments",
        "AMENDMENTS",
        "Repeals",
    ]
    for marker in cutoff_markers:
        if marker in text:
            text = text.split(marker)[0]

    # 5. 공백 및 특수문자 정리
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lstrip(" ,.-:;")

    # 6. 내용 검증: 너무 짧거나(20자 미만), "Repealed"(폐지됨)만 있는 경우 버림
    if len(text) < 20 or "Repealed" in text[:50]:
        return ""

    return text


def get_title(text):
    """미국 법전(US Code) 특화 제목 추출기"""

    # 패턴 1: "Title 51—NATIONAL..." 형태
    # 대시(-)가 여러 종류일 수 있어서 \W로 처리
    match_title = re.search(r"Title\s+\d+\W+([A-Z\s\-\,]+)", text)
    if match_title:
        title_candidate = match_title.group(1).strip()
        # 제목이 5자 이상, 100자 미만일 경우 채택
        if 5 < len(title_candidate) < 100:
            return title_candidate

    # 패턴 2: "cited as"
    match_cited = re.search(
        r'cited as the\s+["\']([^"\']{3,100})["\']', text, re.IGNORECASE
    )
    if match_cited:
        return clean_text(match_cited.group(1))

    # 패턴 3: 첫 줄이 대문자 덩어리인 경우
    first_line = text.split("\n")[0].strip()
    if first_line.isupper() and len(first_line) > 5:
        # Act, Code, Program 등이 포함되어 있으면 제목으로 간주
        if any(k in first_line for k in ["ACT", "CODE", "PROGRAM", "LAW"]):
            return first_line

    return None


def is_historical_noise(text):
    """역사 사료 필터"""
    preview = text[:500].lower()
    noise_keywords = [
        "john adams",
        "abigail adams",
        "george washington",
        "letter to",
        "diary of",
        "obidient servant",
    ]
    return any(k in preview for k in noise_keywords)


def split_into_articles(text):
    chunks = []
    # 패턴: Section 기호(§), Sec., Section, 또는 Article
    pattern = r"((?:Section|Sec\.|§|Article)\s*\d+[a-zA-Z0-9\(\)\-]*)"

    parts = re.split(pattern, text)

    curr_art = "Preamble"  # 조항 번호가 없는 앞부분

    for p in parts:
        p = p.strip()
        if not p:
            continue

        # 조항 번호인지 확인
        if re.match(pattern, p):
            curr_art = p
        else:
            # 내용 부분 처리
            cleaned_content = clean_text(p)
            if cleaned_content:
                chunks.append({"article_no": curr_art, "content": cleaned_content})

    # 만약 쪼개진 게 없는데 내용이 있다면 통째로 저장
    if not chunks and clean_text(text):
        chunks.append({"article_no": "Full Text", "content": clean_text(text)})

    return chunks


def save_data(data):
    rows = []
    text = data["text"]
    country = data["country"]

    # 제목 추출 시도
    title = get_title(text)

    # 제목을 못 찾았으면 "US Code Title [숫자]" 형식으로 저장
    if not title and country == "US":
        match_num = re.search(r"Title\s+(\d+)", text)
        if match_num:
            title = f"US Code Title {match_num.group(1)}"
        else:
            title = "US Federal Law"
    elif not title:
        title = f"{country} Legal Document"

    # 본문 청소
    cleaned_full_text = clean_text(text)

    # 전처리 후 내용 비어있으면 저장X
    if not cleaned_full_text:
        return []

    for chunk in split_into_articles(cleaned_full_text):
        rows.append(
            {
                "country_id": COUNTRY_MAP[country],
                "law_title": clean_text(title),
                "category": "Statute",
                "article_no": clean_text(chunk["article_no"]),
                "content": clean_text(chunk["content"]),
                "enactment_date": "2020-01-01",
                "amendment_date": "2024-01-01",
            }
        )
    return rows


def main():
    final_data = []
    print("법률 데이터 정제 및 수집 V2.0")

    # 1. 미국 데이터 (pile-of-law / uscode)
    print("\n🇺🇸 [US] 수집 중...")
    try:
        us_ds = load_dataset(
            "pile-of-law/pile-of-law",
            "uscode",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        count = 0
        for item in us_ds:
            if count >= TARGET_PER_COUNTRY:
                break
            if len(item["text"]) < 200:
                continue  # 너무 짧은 건 버림

            rows = save_data({"text": item["text"], "country": "US"})
            final_data.extend(rows)
            count += 1
            if count % 50 == 0:
                print(f"   Running... {count} (Sample Title: {rows[0]['law_title']})")
    except Exception as e:
        print(f"US Error: {e}")

    # 2. 캐나다 데이터 (Multi_Legal_Pile)
    print("\n🇨🇦 [CA] 수집 중...")
    try:
        ca_ds = load_dataset(
            "joelniklaus/Multi_Legal_Pile",
            "en_legislation",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        ca_ds = ca_ds.shuffle(seed=42, buffer_size=30000)

        count = 0
        for item in ca_ds:
            if count >= TARGET_PER_COUNTRY:
                break

            text = item.get("text", "")
            jurisdiction = str(item.get("jurisdiction", "")).upper()

            # 캐나다 + 역사 편지 아님
            if (
                "CANADA" in jurisdiction or "CA" in jurisdiction
            ) and not is_historical_noise(text):
                rows = save_data({"text": text, "country": "CA"})
                final_data.extend(rows)
                count += 1
                if count % 50 == 0:
                    print(
                        f"   Running... {count} (Sample Title: {rows[0]['law_title']})"
                    )

    except Exception as e:
        print(f"CA Error: {e}")

    print(f"\n저장 중... 총 {len(final_data)}개 행")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print("완료!")


if __name__ == "__main__":
    main()
