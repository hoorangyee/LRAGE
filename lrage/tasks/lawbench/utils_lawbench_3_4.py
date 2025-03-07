import re
import math

import cn2an

def doc_to_target_3_4(doc) -> str:
    answer = doc["answer"].replace("刑期:", "")
    answer = answer.replace("个月", "")
    
    return answer

def lawbench_3_4_log_distance(predictions, references):
    
    prediction = predictions[0]
    reference = references[0]
    if "死刑" in reference or "无期" in reference:
        # TODO: data imperfection
        return 0.0
    reference = float(reference)

    score_list, abstentions = [], 0

    prediction = cn2an.transform(prediction, "cn2an")
    
    prediction_digit_month_list = re.findall(r"\d+个月", prediction)
    prediction_digit_month_list = [int(digit.replace("个月", "")) for digit in prediction_digit_month_list]
    prediction_digit_month_list2 = re.findall(r"\d+月", prediction)
    prediction_digit_month_list2 = [int(digit.replace("月", "")) for digit in prediction_digit_month_list2]
    prediction_digit_month_list.extend(prediction_digit_month_list2)
    # catches the digits before "年"
    prediction_digit_year_list = re.findall(r"\d+年", prediction)
    prediction_digit_year_list = [int(digit.replace("年", "")) for digit in prediction_digit_year_list]

    if len(prediction_digit_month_list) > 0:
        prediction_digit_month = int(prediction_digit_month_list[0])
    elif len(prediction_digit_year_list) > 0:
        prediction_digit_month = int(prediction_digit_year_list[0]) * 12
    else:
        abstentions += 1
        prediction_digit_month = -1

    if prediction_digit_month != -1:
        score_list.append(abs(math.log(reference + 1) - math.log(prediction_digit_month + 1)))
    else:
        score_list.append(math.log(216))

    # compute the average of score_list (log distance)
    log_distance = sum(score_list) / len(score_list)
    # normalize the score to between 0 and 1
    log_distance = (math.log(216) - log_distance)/math.log(216)
    return log_distance