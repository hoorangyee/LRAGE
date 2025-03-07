def doc_to_target_3_6(doc) -> str:
    answer_letter = doc["answer"][5]
    return answer_letter

def multi_choice_judge_3_6(prediction, option_list, answer_token):
    # a dict, key: letters in the option list, value: count of the letter in the prediction
    count_dict, abstention, accuracy = {}, 0, 0
    for option in option_list:
        option_count = prediction.count(option)
        count_dict[option] = 1 if option_count > 0 else 0  # multiple occurrence of the same letter is counted as 1

    if sum(count_dict.values()) == 0:
        abstention = 1
    # if the answer token is the only predicted token, the prediction is correct 
    elif count_dict[answer_token] == 1 and sum(count_dict.values()) == 1:
        accuracy = 1
    return {"score": accuracy, "abstention": abstention}

def judge_mc_3_6(predictions, references):
    prediction = predictions[0]
    reference = references[0]

    option_list = ["A", "B", "C", "D"]
    judge = multi_choice_judge_3_6(prediction, option_list, reference)
    return judge["score"]
