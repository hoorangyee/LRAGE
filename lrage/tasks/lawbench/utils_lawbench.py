from sklearn.metrics import f1_score

def doc_to_text(doc) -> str:
    return doc["question"]