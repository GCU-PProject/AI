import json
import time
import requests
from lxml import etree

MASTER_ACTS_CANADA = {
    "PEN": "C-46",    # Criminal Code
    "LAB": "L-2",     # Canada Labour Code
    "IMM": "I-2.5",   # Immigration and Refugee Protection Act
    "VEH": "M-10.01", # Motor Vehicle Safety Act
    "HSC": "H-6",     # Canada Health Act
    "EDC": "S-22.7",  # Canada Student Financial Assistance Act (좀 더 양이 많은 법령으로 교체)
    "CIV": "C-11",    # Civil Marriage Act
    "INS": "I-11.8",  # Insurance Companies Act
    "GOV": "F-11"     # Financial Administration Act
}

BASE_XML_URL = "https://laws-lois.justice.gc.ca/eng/XML"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def fetch_canada_ultra_deep(act_id, law_code):
    target_url = f"{BASE_XML_URL}/{act_id}.xml"
    print(f"📡 {law_code} 수집 중... ({target_url})")
    
    try:
        res = requests.get(target_url, headers=HEADERS)
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(res.content, parser=parser)
        
        title_node = root.find(".//LongTitle")
        category_title = title_node.text if title_node is not None else act_id
        
        results = []

        items = root.xpath(".//Subsection | .//Paragraph")
        
        if not items:
            items = root.findall(".//Section")

        for item in items:
            # 1. 조항 번호 추적
            parent_section = item.xpath("./ancestor::Section[1]/Label/text()")
            section_no = parent_section[0] if parent_section else ""
            
            # 현재 요소의 번호 (항/호 번호)
            current_label = item.find("Label")
            label_text = current_label.text if current_label is not None else ""
            
            # 최종 번호 형식: Section 230(1)(a)
            full_article_no = f"Section {section_no}({label_text})" if section_no else label_text

            # 2. 본문 텍스트 추출
            texts = item.xpath(".//Text//text()")
            content_text = " ".join([t.strip() for t in texts if t.strip()])
            
            if content_text:
                results.append({
                    "law_code": law_code,
                    "category": category_title.strip(),
                    "article_no": full_article_no,
                    "content": content_text,
                    "url": f"https://laws-lois.justice.gc.ca/eng/acts/{act_id}/FullText.html"
                })
        
        return results
    except Exception as e:
        print(f"  🚨 에러 발생: {e}")
        return []

# ============== 메인 실행부 ==============

if __name__ == "__main__":
    output_file = "Canada_Law_Data_Ultra_Deep.jsonl"
    map_id_counter = 1
    
    print("🚀 캐나다 연방법 데이터 증폭 수집 시작")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for law_code, act_id in MASTER_ACTS_CANADA.items():
            data = fetch_canada_ultra_deep(act_id, law_code)
            
            if data:
                print(f"  ✅ {law_code}: {len(data)}개 세부 항목 수집 완료")
                for item in data:
                    row = {
                        "map_id": map_id_counter,
                        **item
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    map_id_counter += 1
            time.sleep(0.5)

    print(f"\n✨ 완료! 총 {map_id_counter-1}개의 데이터가 저장되었습니다.")