import json
from datasets import load_dataset
import re

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def classify_law_code(text, title):
    """
    텍스트와 제목 키워드를 기반으로 캘리포니아 법령 코드 형식처럼 해당 법률의 주제를 분류
    """
    title_lower = (title or "").lower()
    text_lower = str(text).lower()[:1000] # 최적화를 위해 앞부분 위주로 검사
    combined = f"{title_lower} {text_lower}"
    
    # 1. VEH (Vehicle Code - 교통법)
    if any(k in combined for k in ["vehicle", "traffic", "transport", "road", "driving", "motor", "highway"]):
        return "VEH"
        
    # 2. PEN (Penal Code - 형법)
    if any(k in combined for k in ["penal", "criminal", "offense", "felony", "misdemeanor", "punishment", "prison", "sentence", "crime"]):
        return "PEN"
        
    # 3. CIV (Civil Code / Civil Procedure - 민법 및 민사소송법)
    if any(k in combined for k in ["civil", "tort", "contract", "liability", "damages", "negligence", "obligation"]):
        return "CIV"
        
    # 4. FAM (Family Code - 가족법)
    if any(k in combined for k in ["family", "marriage", "divorce", "custody", "adoption", "domestic", "child", "parent"]):
        return "FAM"
        
    # 5. LAB (Labor Code - 노동법)
    if any(k in combined for k in ["labor", "employment", "workplace", "employee", "employer", "wage", "industrial", "worker"]):
        return "LAB"
        
    # 6. HSC (Health and Safety Code - 보건안전법)
    if any(k in combined for k in ["health", "medical", "hospital", "patient", "safety", "disease", "public health", "drug"]):
        return "HSC"
        
    # 7. EDC (Education Code - 교육법)
    if any(k in combined for k in ["education", "school", "student", "teacher", "university", "college", "academic"]):
        return "EDC"
        
    # 8. FIN (Financial Code - 금융법)
    if any(k in combined for k in ["financial", "bank", "loan", "credit", "finance", "investment", "insurance"]):
        return "FIN"
        
    # 9. GOV (Government Code - 행정/정부법)
    if any(k in combined for k in ["government", "administrative", "council", "commission", "state", "public service", "department"]):
        return "GOV"
        
    # 10. ENV (Environment/Public Resources - 환경/자원법)
    if any(k in combined for k in ["environment", "pollution", "conservation", "water", "nature", "wildlife", "climate"]):
        return "ENV"
        
    # 11. HOU (Housing / Real Estate - 주거/부동산법)
    if any(k in combined for k in ["housing", "real estate", "property", "tenant", "landlord", "lease", "rent", "building"]):
        return "HOU"
        
    # 매칭되는 것이 없으면 기본값 (일반 법률)
    return "GEN"

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
    map_id_counter = 1
    
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
            
            # 주제별 분류 코드 생성
            law_code = classify_law_code(text, section_title)
            
            sections = split_into_articles(text)
            for sec in sections:
                row = {
                    "map_id": map_id_counter,
                    "country_id": country_id,  # laws 테이블의 country_id 필드와 조인될 수 있도록 변경
                    "law_code": law_code,
                    "category": section_title,
                    "article_no": sec["article_no"],
                    "content": sec["content"],
                    "url": url
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                map_id_counter += 1

    print(f"✅ '{output_filename}' 파일이 생성되었습니다. (총 {map_id_counter-1}개 조항 매핑 완료)")

if __name__ == "__main__":
    export_to_jsonl()