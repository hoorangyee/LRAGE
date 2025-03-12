def doc_to_text(doc) -> str:
    return "\n".join([doc["instruction"], doc["question"]])