def doc_to_text_essay(doc) -> str:
    return doc["question"]

def doc_to_target_essay(doc) -> str:
    return doc["answer"]

def doc_to_rubric_essay(doc)->str:
    return doc["rubric"]

def doc_to_text_mc(doc) -> str:
    text = f"당신은 사용자의 질문에 친절하고 논리적으로 답변해 주는 세무 전문가 챗봇입니다. 위 제시된 가산세 부과처분의 배경을 읽고 해당 가산세 부과에 대해, \"적법함\", \"적법하지 않음\" 둘 중 하나를 선택하여 답변하고 설명해주세요. 답을 내릴 수 없다면 \"알 수 없음\"으로 답변하고 설명해주세요.:\n{doc['case_info']}\n{doc['facts']}\n{doc['plaintiff_claims']}\n{doc['defendant_claims']}"
    return text

def doc_to_target_mc(doc) -> str:
    target = "적법함" if doc["laufulness"] else "적법하지 않음"
    return target