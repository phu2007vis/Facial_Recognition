questions = [
            "nghiem tuc",
            "smile"
            ]
def question_bank(index):
    return questions[index]
#  "blink eyes"

def challenge_result(question, out_model):
    # out_model = out_model[1]
    if question == "smile":
        if len(out_model["emotion"]) == 0:
            challenge = "fail"
        elif out_model["emotion"][0] == "happy": 
            challenge = "pass"
        else:
            challenge = "fail"

    elif question == "nghiem tuc":
        if len(out_model["emotion"]) == 0:
            challenge = "fail"
        elif out_model["emotion"][0] == "neutral": 
            challenge = "pass"
        else:
            challenge = "fail"
        
    return challenge