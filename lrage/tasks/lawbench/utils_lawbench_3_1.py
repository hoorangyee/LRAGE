import re
import cn2an

def doc_to_target_3_1(doc) -> str:
    answer = doc["answer"].replace("法条:刑法第", "")
    answer = answer.replace("条", "")
    answer_law_indices = answer.split("、")

    return answer_law_indices

def replace_match(match):
    return match.group(1)

def lawbench_3_1_f1_score(predictions, references):
    prediction_law_chunks = predictions[0].split("、")
    prediction_law_index_digit_list = []

    for prediction_law_chunk in prediction_law_chunks:
        prediction_law_chunk = prediction_law_chunk.replace("万元", "元")

        # delete phrase starts with "第" and ends with "款", we don't care about it in the answer
        prediction_law_chunk = re.sub(r'第(.*?)款', "", prediction_law_chunk)
        # keep only the digits in the phrase starts with "第" and ends with "条", otherwise cn may fail to convert
        prediction_law_chunk = re.sub(r'第(.*?)条', replace_match, prediction_law_chunk)
        prediction_law_chunk = cn2an.transform(prediction_law_chunk, "cn2an")
        # find digtis in prediction_law_chunk
        prediction_law_section_numbers = re.findall(r"\d+", prediction_law_chunk)
        if len(prediction_law_section_numbers) == 0:
            continue
        if len(prediction_law_section_numbers) != 1:
            # in this case, we only take the first number, and reject the others
            pass

        prediction_law_index_digit = int(prediction_law_section_numbers[0])
        prediction_law_index_digit_list.append(prediction_law_index_digit)
    
    gt_set = set(map(int, references))
    pred_set = set(prediction_law_index_digit_list)

    precision = len(gt_set.intersection(pred_set)) / len(pred_set) if len(pred_set) != 0 else 0
    recall = len(gt_set.intersection(pred_set)) / len(gt_set) if len(gt_set) != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return f1_score