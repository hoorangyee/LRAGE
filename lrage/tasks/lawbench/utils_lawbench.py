from sklearn.metrics import f1_score

def doc_to_text(doc) -> str:
    return doc["question"]

def lawbench_3_3_f1_score(predictions, references):
    fscore = f1_score(references, predictions, average="macro")
    return fscore