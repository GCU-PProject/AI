import json
from datasets import load_dataset
import re

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_articles(text):
    chunks = []
    # 호주 법률 분리를 위한 정규식 패턴
    pattern = r'((?:Section|Sec\.|§|s\.|Part|Pt\.)\s*\d+[a-zA-Z0-9\(\)\-\.]*)'
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    curr_art = None
    
    for p in parts:
        p = p.strip()
        if not p: continue
        
        if re.match(pattern, p, flags=re.IGNORECASE): 
            normalized = re.sub(r'\s+', ' ', p)
            curr_art = normalized
        else:
            cleaned_content = clean_text(p)
            if cleaned_content and len(cleaned_content) >= 50:
                if not curr_art:
                    curr_art = "General Provisions"
                chunks.append({"article_no": curr_art, "content": cleaned_content})
    
    if not chunks:
        cleaned_full = clean_text(text)
        if cleaned_full and len(cleaned_full) >= 50:
            chunks.append({"article_no": "Full Text", "content": cleaned_full})
        
    return chunks

def export_to_jsonl():
    print("📊 데이터셋 로딩 중...")
    ds = load_dataset("isaacus/open-australian-legal-corpus")
    
    # 상위 1,000개 테스트 데이터 뽑기
    corpus = ds['corpus'].select(range(1000))
    print("🔄 텍스트 분류 및 law_data JSONL 형식 매핑 중...")
    
    output_filename = "law_data_AU.jsonl"
    processed_count = 0
    
    # 호주 관할권별 임의의 country_id 매핑 딕셔너리
    jurisdiction_map = {
        'commonwealth': 2,        # 호주 연방
        'new_south_wales': 3,     # 뉴사우스웨일스
        'queensland': 4,          # 퀸즐랜드
        'western_australia': 5,   # 서호주
        'south_australia': 6,     # 남호주
        'tasmania': 7,            # 태즈메이니아
        'norfolk_island': 8       # 노퍽 섬
    }
    
    with open(output_filename, "w", encoding="utf-8") as f_out:
        for item in corpus:
            text = item.get('text', '')
            url = item.get('url', '')
            section_title = item.get('citation', '')
            jurisdiction = item.get('jurisdiction', 'unknown')
            
            # 딕셔너리에서 country_id 값을 매핑 (매칭 안 되면 기본값 9)
            country_id = jurisdiction_map.get(jurisdiction, 9) 
            
            sections = split_into_articles(text)
            for sec in sections:
                row = {
                    "country_id": country_id,  # laws 테이블의 country_id 필드와 조인
                    "law_type": "",
                    "section_title": section_title,
                    "article_no": sec["article_no"],
                    "content": sec["content"],
                    "source_url": url
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                processed_count += 1

    print(f"✅ '{output_filename}' 파일이 생성되었습니다. (총 {processed_count}개 조항 매핑 완료)")

if __name__ == "__main__":
    export_to_jsonl()