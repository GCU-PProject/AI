import json
import re
import os
import gc
from datetime import datetime
from datasets import load_dataset

# ================= 설정 =================
TARGET_PER_COUNTRY = 500

# 메모리 에러 방지
SHUFFLE_BUFFER_SIZE = 2000 

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 윈도우 경로 호환성을 위해 normpath 사용
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'data'))
OUTPUT_FILE = os.path.join(DATA_DIR, 'laws_data.json')

COUNTRY_MAP = {'US': 1, 'CA': 2}

# 최소 컨텐츠 길이 (너무 짧은 조각 필터링)
MIN_CONTENT_LENGTH = 50

# ================= 로직 =================


def clean_text(text):
    if not text:
        return ""

    # 1. 시스템 헤더 및 변환기 로그 제거
    text = re.sub(r'.*?Online@[\w\-]+\s+(yes|no)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'.*?USCConverter\s+[\d\.]+', '', text, flags=re.IGNORECASE)

    # 2. 불필요한 헤더/푸터 제거
    text = re.sub(r'Title\s+\d+\s+USC.*', '', text)
    text = re.sub(r'Current through.*', '', text)
    text = re.sub(r'OLRC\s+\d{4}-\d{2}-\d{2}T[\d:]+', '', text)  # OLRC 타임스탬프 제거

    # 3. 목차 제거
    if len(re.findall(r'\d{3,}\.[A-Z]', text)) > 3: 
        return ""

    # 4. 개정 이력 자르기
    cutoff_markers = [
        "Editorial Notes", "Statutory Notes", "Historical and Revision",
        "Amendments", "AMENDMENTS", "Repeals", "References in Text",
        "Transfer of Functions", "Effective Date"
    ]
    for marker in cutoff_markers:
        if marker in text:
            text = text.split(marker)[0]

    # 5. Pub. L. 같은 메타데이터 제거 (하지만 날짜는 보존)
    # 단독으로 있는 Pub. L. 라인 제거
    text = re.sub(r'^Pub\.\s*L\.\s+[\d–\-]+\s*,?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d{1,2}\s+Stat\.\s+\d+[,\).]?\s*$', '', text, flags=re.MULTILINE)
    
    # 6. 공백 및 특수문자 정리
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lstrip(' ,.-:;')

    # 7. 내용 검증
    if len(text) < MIN_CONTENT_LENGTH or "Repealed" in text[:50]:
        return ""
    
    # 8. 의미 없는 조각 필터링 (날짜만 있거나 Pub. L.만 있는 경우)
    if re.match(r'^[A-Z][a-z]+\s+\d{1,2},\s+\d{4}.*$', text) and len(text) < 100:
        # 날짜로 시작하는 짧은 텍스트는 대부분 메타데이터
        if not any(keyword in text.lower() for keyword in ['act', 'law', 'code', 'section', 'shall', 'may', 'must']):
            return ""

    return text


def get_title(text):
    # 1. "Short title" 패턴: "This Act may be cited as the '...'"
    match_short = re.search(r'(?:Short title|This Act may be cited as|cited as the)\s+["\']([^"\']{5,150})["\']', text, re.IGNORECASE)
    if match_short:
        title = match_short.group(1).strip()
        if 5 < len(title) < 150:
            return title
    
    # 2. Title 번호와 함께 있는 제목
    match_title = re.search(r'Title\s+\d+\s*[—–-]?\s*([A-Z][A-Z\s\-\,\(\)]+?)(?:\s+\[|$|USC|APPENDIX)', text)
    if match_title:
        title_candidate = match_title.group(1).strip()
        # 너무 짧거나 긴 경우 필터링
        if 5 < len(title_candidate) < 100:
            return title_candidate
    
    # 3. "cited as" 패턴
    match_cited = re.search(r'cited as\s+["\']([^"\']{5,100})["\']', text, re.IGNORECASE)
    if match_cited:
        title = match_cited.group(1).strip()
        if 5 < len(title) < 100:
            return title
    
    # 4. 대문자로 시작하는 첫 줄 (법률명 패턴)
    lines = text.split('\n')
    for line in lines[:5]:  # 처음 5줄만 확인
        line = line.strip()
        if not line:
            continue
        # 대문자로 시작하고 ACT, CODE, PROGRAM, LAW, AGREEMENT 등 포함
        if (line[0].isupper() and len(line) > 10 and len(line) < 200):
            if any(k in line.upper() for k in ["ACT", "CODE", "PROGRAM", "LAW", "AGREEMENT", "TREATY", "REGULATION"]):
                # 너무 많은 특수문자 제외
                if line.count('§') < 3 and line.count('(') < 5:
                    return line
    
    # 5. Title 번호만 있는 경우
    match_num = re.search(r'Title\s+(\d+)', text)
    if match_num:
        return f"US Code Title {match_num.group(1)}"
    
    return None


def is_historical_noise(text):
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

def classify_category(text, title, country):
    """법률 카테고리 자동 분류"""
    text_lower = text.lower()
    title_lower = (title or "").lower()
    combined = f"{text_lower} {title_lower}"
    
    # Constitutional Law
    if any(k in combined for k in ["constitution", "constitutional", "amendment", "bill of rights"]):
        return "Constitutional Law"
    
    # Criminal Law
    if any(k in combined for k in ["criminal", "penal", "offense", "felony", "misdemeanor", "punishment", "sentence"]):
        return "Criminal Law"
    
    # Civil Law / Contract
    if any(k in combined for k in ["contract", "tort", "civil", "liability", "damages", "negligence"]):
        return "Civil Law"
    
    # Administrative Law / Regulation
    if any(k in combined for k in ["regulation", "administrative", "agency", "rule", "directive", "order"]):
        return "Administrative Law"
    
    # Tax Law
    if any(k in combined for k in ["tax", "revenue", "irs", "income tax", "taxation"]):
        return "Tax Law"
    
    # Labor Law
    if any(k in combined for k in ["labor", "employment", "workplace", "employee", "employer", "wage"]):
        return "Labor Law"
    
    # Family Law
    if any(k in combined for k in ["family", "marriage", "divorce", "custody", "adoption", "domestic"]):
        return "Family Law"
    
    # Commercial Law
    if any(k in combined for k in ["commercial", "business", "corporation", "trade", "commerce", "merchant"]):
        return "Commercial Law"
    
    # Environmental Law
    if any(k in combined for k in ["environment", "pollution", "epa", "environmental", "conservation"]):
        return "Environmental Law"
    
    # Health Law
    if any(k in combined for k in ["health", "medical", "healthcare", "hospital", "physician", "patient"]):
        return "Health Law"
    
    # Immigration Law
    if any(k in combined for k in ["immigration", "visa", "citizenship", "alien", "naturalization"]):
        return "Immigration Law"
    
    # Agreement / Treaty
    if any(k in combined for k in ["agreement", "treaty", "convention", "protocol", "accord"]):
        return "Agreement"
    
    # 기본값: Statute
    return "Statute"

def extract_dates(text):
    """텍스트에서 날짜 정보 추출"""
    enactment_date = None
    amendment_date = None
    
    # Pub. L. 날짜 패턴: "Pub. L. 91–538, Dec. 9, 1970"
    pub_l_match = re.search(r'Pub\.\s*L\.\s+[\d–\-]+[,\s]+([A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4})', text)
    if pub_l_match:
        date_str = pub_l_match.group(1)
        try:
            # "Dec. 9, 1970" -> "1970-12-09"
            date_obj = datetime.strptime(date_str.replace('.', ''), "%b %d, %Y")
            enactment_date = date_obj.strftime("%Y-%m-%d")
        except:
            pass
    
    # Stat. 날짜 패턴: "June 19, 1968, 82 Stat. 236"
    stat_match = re.search(r'([A-Z][a-z]+\s+\d{1,2},\s+\d{4}),\s+\d+\s+Stat\.', text)
    if stat_match and not enactment_date:
        date_str = stat_match.group(1)
        try:
            date_obj = datetime.strptime(date_str, "%B %d, %Y")
            enactment_date = date_obj.strftime("%Y-%m-%d")
        except:
            pass
    
    # Amendment 날짜 찾기
    amend_patterns = [
        r'amended\s+by\s+Pub\.\s*L\.\s+[\d–\-]+[,\s]+([A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4})',
        r'as\s+amended\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})',
    ]
    for pattern in amend_patterns:
        amend_match = re.search(pattern, text, re.IGNORECASE)
        if amend_match:
            date_str = amend_match.group(1)
            try:
                date_obj = datetime.strptime(date_str.replace('.', ''), "%b %d, %Y")
                amendment_date = date_obj.strftime("%Y-%m-%d")
                break
            except:
                try:
                    date_obj = datetime.strptime(date_str, "%B %d, %Y")
                    amendment_date = date_obj.strftime("%Y-%m-%d")
                    break
                except:
                    pass
    
    return enactment_date, amendment_date

def split_into_articles(text):
    chunks = []
    # 더 다양한 섹션 패턴 인식
    pattern = r'((?:Section|Sec\.|§|§\s*|s\.|s\s+|Article|Art\.|art\.|art\s+|Chapter|Ch\.|ch\.|Part|Pt\.|pt\.)\s*\d+[a-zA-Z0-9\(\)\-\.]*)'
    
    parts = re.split(pattern, text)
    curr_art = None
    
    for p in parts:
        p = p.strip()
        if not p: continue
        
        # 섹션 번호 패턴 매칭
        if re.match(pattern, p): 
            # 정규화: Section 123 -> Section 123
            normalized = re.sub(r'\s+', ' ', p)
            curr_art = normalized
        else:
            cleaned_content = clean_text(p)
            # 최소 길이 체크
            if cleaned_content and len(cleaned_content) >= MIN_CONTENT_LENGTH:
                # article_no가 없으면 자동 생성
                if not curr_art:
                    # 텍스트 시작 부분에서 섹션 번호 찾기
                    section_match = re.search(r'(?:Section|Sec\.|§|s\.|Article|Art\.)\s*(\d+[a-zA-Z0-9\(\)\-\.]*)', cleaned_content[:200])
                    if section_match:
                        curr_art = f"Section {section_match.group(1)}"
                    else:
                        curr_art = "General Provisions"
                
                chunks.append({"article_no": curr_art, "content": cleaned_content})
    
    # 섹션으로 나뉘지 않은 경우 전체 텍스트 사용
    if not chunks:
        cleaned_full = clean_text(text)
        if cleaned_full and len(cleaned_full) >= MIN_CONTENT_LENGTH:
            # 제목이나 섹션 정보 추출 시도
            section_match = re.search(r'(?:Section|Sec\.|§|s\.|Article|Art\.)\s*(\d+[a-zA-Z0-9\(\)\-\.]*)', cleaned_full[:300])
            if section_match:
                article_no = f"Section {section_match.group(1)}"
            else:
                article_no = "Full Text"
            chunks.append({"article_no": article_no, "content": cleaned_full})
        
    return chunks


def save_data(data):
    rows = []
    text = data['text']
    country = data['country']
    
    # 제목 추출
    title = get_title(text)
    
    # 제목이 없으면 기본값 설정
    if not title:
        if country == 'US':
            match_num = re.search(r'Title\s+(\d+)', text)
            if match_num:
                title = f"US Code Title {match_num.group(1)}"
            else:
                # 텍스트에서 법률명 추출 시도
                first_sentence = text.split('.')[0].strip()
                if len(first_sentence) > 10 and len(first_sentence) < 200:
                    title = first_sentence[:150]
                else:
                    title = "US Federal Law"
        else:
            title = f"{country} Legal Document"
    
    # 날짜 추출
    enactment_date, amendment_date = extract_dates(text)
    if not enactment_date:
        enactment_date = "2020-01-01"  # 기본값
    if not amendment_date:
        amendment_date = "2024-01-01"  # 기본값

    cleaned_full_text = clean_text(text)
    
    if not cleaned_full_text or len(cleaned_full_text) < MIN_CONTENT_LENGTH:
        return []

    # 카테고리 분류
    category = classify_category(cleaned_full_text, title, country)

    # 섹션별로 분할
    chunks = split_into_articles(cleaned_full_text)
    
    for chunk in chunks:
        # article_no 정리
        article_no = chunk['article_no']
        if article_no:
            article_no = clean_text(article_no)
        else:
            article_no = "General Provisions"
        
        # content 검증
        content = chunk['content']
        if not content or len(content) < MIN_CONTENT_LENGTH:
            continue
        
        rows.append({
            "country_id": COUNTRY_MAP[country],
            "law_title": clean_text(title) if title else f"{country} Legal Document",
            "category": category,
            "article_no": article_no,
            "content": content,
            "enactment_date": enactment_date,
            "amendment_date": amendment_date
        })
    
    return rows


def main():
    global OUTPUT_FILE
    
    final_data = []
    print("🚀 법률 데이터 정제 및 수집 V2.1 (메모리 최적화)")
    print(f"📂 저장 경로: {OUTPUT_FILE}")

    # data 폴더 생성
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
            print(f"   📂 '{DATA_DIR}' 폴더를 생성했습니다.")
        except OSError as e:
            print(f"   ⚠️ 폴더 생성 실패 (권한 문제 등): {e}")
            # 폴더 생성 실패시 현재 폴더에 저장 시도
    
            OUTPUT_FILE = os.path.join(BASE_DIR, 'laws_data.json')
            print(f"   ↳ 대신 현재 폴더에 저장합니다: {OUTPUT_FILE}")

    # 1. 미국 데이터
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
            if count >= TARGET_PER_COUNTRY: break
            if len(item['text']) < 200: continue
            
            rows = save_data({'text': item['text'], 'country': 'US'})
            final_data.extend(rows)
            count += 1
            if count % 50 == 0:
                print(f"   Running... {count} (Sample Title: {rows[0]['law_title']})")
                gc.collect() # 주기적으로 메모리 청소
    except Exception as e: print(f"❌ US Error: {e}")

    # 2. 캐나다 데이터
    print("\n🇨🇦 [CA] 수집 중...")
    print(f"   👉 메모리 보호를 위해 셔플 버퍼를 {SHUFFLE_BUFFER_SIZE}개로 제한합니다.")
    try:
        ca_ds = load_dataset("joelniklaus/Multi_Legal_Pile", "en_legislation", split="train", streaming=True, trust_remote_code=True)
        
        # [수정] 버퍼 사이즈를 대폭 줄임 (30000 -> 2000)
        ca_ds = ca_ds.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER_SIZE)
        
        count = 0
        for item in ca_ds:
            if count >= TARGET_PER_COUNTRY: break
            
            try:
                text = item.get('text', '')
                jurisdiction = str(item.get('jurisdiction', '')).upper()
                
                if ("CANADA" in jurisdiction or "CA" in jurisdiction) and not is_historical_noise(text):
                    rows = save_data({'text': text, 'country': 'CA'})
                    final_data.extend(rows)
                    count += 1
                    if count % 50 == 0: 
                        print(f"   Running... {count} (Sample Title: {rows[0]['law_title']})")
                        gc.collect() # 주기적으로 메모리 청소
            except Exception:
                # 특정 데이터가 너무 커서 에러나면 건너뛰기
                continue

    except Exception as e: print(f"❌ CA Error: {e}")

    print(f"\n💾 저장 중... 총 {len(final_data)}개 행")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print("🎉 완료!")


if __name__ == "__main__":
    main()