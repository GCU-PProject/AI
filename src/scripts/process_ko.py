import requests
import json
import time
import re

auth_key = "4588"
search_url = f"https://www.law.go.kr/DRF/lawSearch.do?OC={4588}&target=law&type=JSON&display=100"

def classify_ko_law_type_by_ministry(ministry_name):
    """소관부처명 기준으로 매핑하며, 불필요한 부처는 예외 없이 싹 다 제외(SKIP) 처리"""
    if not ministry_name: return 'SKIP'
    first_ministry = ministry_name.split(',')[0].strip()
    
    # 1. 수집하지 않아도 되는 부처 필터링
    skip_keywords = [
        '국토교통', '공정거래', '행정안전', '기획재정', '금융위', 
        '인사혁신', '법제처', '국가교육', '교육부', '선거관리',
        '국방', '병무', '방위', '농림', '농촌', '산림', 
        '유산', '문화', '체육', '관광', '환경', '산업통상', 
        '과학기술', '중소벤처', '해양수산', '보훈', '통일', '여성가족'
    ]
    for skip in skip_keywords:
        if skip in first_ministry: return 'SKIP'
            
    mapping = {
        '국토교통부': 'VEH',
        '교육부': 'EDC', '국가교육위원회': 'EDC',
        '고용노동부': 'LAB', '최저임금위원회': 'LAB',
        '경찰청': 'PEN', '검찰청': 'PEN',
        '보건복지부': 'HSC', '식품의약품안전처': 'HSC', '질병관리청': 'HSC',
        '행정안전부': 'GOV', '법제처': 'GOV', '주민등록': 'GOV',
        '외교부': 'FOR', '재외동포청': 'FOR', # 여권, 영사, 재외국민 등록
        '법무부': 'LGL', # 비자, 국적, 출입국, 기본 형법/민법
        '대법원': 'JUD', '헌법재판소': 'JUD', # 가족관계등록(혼인, 출생), 소송
        '국세청': 'TAX', '관세청': 'TAX', # 세금, 관세
        '기획재정부': 'ECO', '금융위원회': 'ECO' # 금융, 세제, 외환
    }
    
    for key, code in mapping.items():
        if key in first_ministry: return code
            
    return 'SKIP'

def format_date(date_str):
    if not date_str or len(date_str) != 8:
        return None
    try:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    except:
        return None

def extract_article_no(content_str):
    content_str = str(content_str)
    match = re.match(r'^(제\d+조(?:의\d+)?)\s*(?:\([^)]+\))?', content_str)
    if match: return match.group(1)
    if content_str.startswith('부칙'): return '부칙'
    return 'General Provisions'

def build_article_content(jomun):
    content = jomun.get('조문내용', '')
    if isinstance(content, list): content = "\n".join(str(c) for c in content)
    content = str(content).strip()
    
    # 1. 항 (Paragraphs)
    hangs = jomun.get('항', [])
    if isinstance(hangs, dict): 
        hangs = [hangs]

    if isinstance(hangs, list):
        for hang in hangs:
            hang_content = hang.get('항내용', '').strip()
            if hang_content:
                content += "\n" + hang_content
            
            # 2. 호 (Subparagraphs)
            hos = hang.get('호', [])
            if isinstance(hos, dict): hos = [hos]
            if isinstance(hos, list):
                for ho in hos:
                    ho_content = ho.get('호내용', '').strip()
                    if ho_content:
                        content += "\n  " + ho_content
                    # 3. 목 (Items)
                    moks = ho.get('목', [])
                    if isinstance(moks, dict): moks = [moks]
                    if isinstance(moks, list):
                        for mok in moks:
                            mok_content = mok.get('목내용', '').strip()
                            if mok_content:
                                content += "\n    " + mok_content
    
    hos_direct = jomun.get('호', [])
    if isinstance(hos_direct, dict): hos_direct = [hos_direct]
    if isinstance(hos_direct, list):
        for ho in hos_direct:
            ho_content = ho.get('호내용', '').strip()
            if ho_content:
                content += "\n  " + ho_content
                
    return content

def main():
    print("목록 전체 조회 중... (페이지네이션 적용하여 전체 법령 긁어옵니다)")
    all_laws = []
    
    # 5600개 이상이므로 넉넉히 60페이지까지 탐색 (한 줄당 100개씩 최대 6000개 수집)
    MAX_PAGES = 60
    
    for page in range(1, MAX_PAGES + 1):
        search_url = f"https://www.law.go.kr/DRF/lawSearch.do?OC={auth_key}&target=law&type=JSON&display=100&page={page}"
        print(f"👉 법령 목록 {page}페이지 로딩 중...")
        
        try:
            response = requests.get(search_url, timeout=10)
            if response.status_code != 200:
                break
                
            data = response.json()
            laws = data.get('LawSearch', {}).get('law', [])
            
            if not laws:
                break # 더 이상 데이터가 없으면 탈출
                
            all_laws.extend(laws)
            time.sleep(0.1) # 서버 부하 방지
            
        except Exception as e:
            print(f"페이지 로딩 에러 ({page}p): {e}")
            break
            
    print(f"✅ 법제처 법령 목록 수집 완료: 총 {len(all_laws)}개의 전체 법령 분석 돌입!")
    
    output_filename = "law_data_KO.jsonl"
    processed_count = 0
    skipped_count = 0
    
    print("데이터 필터링 및 🔎조문별 본문 상세 수집 시작 (시간이 꽤 소요됩니다)")
    with open(output_filename, 'w', encoding='utf-8') as f_out:
        for idx, law in enumerate(all_laws):
            title = law.get('법령명한글', '')
            ministry = law.get('소관부처명', '')
            law_type = classify_ko_law_type_by_ministry(ministry)
            
            if law_type == 'SKIP':
                skipped_count += 1
                continue
            
            mst = law.get('법령일련번호')
            if not mst: 
                continue
                
            print(f"[{idx+1}/{len(all_laws)}] 본문 수집 중: {title} ({law_type})")
            
            detail_url = f"https://www.law.go.kr/DRF/lawService.do?OC={auth_key}&target=law&MST={mst}&type=JSON"
            
            try:
                res = requests.get(detail_url, timeout=10)
                if res.status_code != 200: continue
                detail_data = res.json()
                
                jomun_list = detail_data.get('법령', {}).get('조문', {}).get('조문단위', [])
                if isinstance(jomun_list, dict): jomun_list = [jomun_list]
                
                if not jomun_list: continue
                
                for jomun in jomun_list:
                    # ✅ 여기서 list 파싱 버그 방어 완료
                    raw_content = build_article_content(jomun)
                    if not raw_content or len(raw_content) < 5:
                        continue
                        
                    article_no = extract_article_no(raw_content)
                    
                    enactment_date = format_date(law.get('시행일자', ''))
                    amendment_date = format_date(law.get('공포일자', ''))
                    
                    detail_link = law.get('법령상세링크', '')
                    source_url = f"https://www.law.go.kr{detail_link}" if detail_link else ""
                    
                    row = {
                        "country_id": 9,
                        "law_type": law_type,
                        "section_title": law.get('법령구분명', ''),
                        "article_no": article_no,
                        "content": f"[{title}] {raw_content}",
                        "source_url": source_url,
                        "enactment_date": enactment_date,
                        "amendment_date": amendment_date
                    }
                    
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    processed_count += 1
                    
                f_out.flush() # 저장 안정성을 위해 주기적으로 파일에 쓰기
                    
            except Exception as e:
                print(f"상세 수집 실패 ({title}): {e}")
                
            time.sleep(0.3) # API 제한 안 걸리게 약간의 휴식
            
    print(f"\n✅ 전체 수집 및 처리 완료!")
    print(f"📥 수집된 실제 조문(Article) 데이터: 총 {processed_count}개")
    print(f"🗑️ 쓰레기값이라 버려진 법령: 총 {skipped_count}개")
    print(f"💾 데이터가 '{output_filename}' 에 저장되었습니다.")

if __name__ == "__main__":
    main()