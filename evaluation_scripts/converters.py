import os
import json


# Code taken from
# https://gitlab.com/hwu-ilab/hermit-nlu/blob/master/data/nlu_benchmark/
# to ensure same evaluation metrics as https://arxiv.org/abs/1910.00912
# Additional methods were added to be able to call desired evaluation methods.


arg_format_pattern = r"\[\s*(?P<label>[\w']*)\s*:(?P<filler>[\s\w'\.@\-&+]+)\]"
arg_annotation_pattern = r"\[\s*[\w']*\s*:[\s\w'\.@\-&+]+\]"


def load_json_prediction_file(predictions_file):
    _, filename = os.path.split(predictions_file)
    print ("Loading {}..".format(filename))
    with open(predictions_file, "r") as f:
        json_prediction = json.load(f)
        f.close()
    return json_prediction


def squeeze_prediction_span(json_prediction):
    squeezed_predictions = []
    for example in json_prediction:
        new_example = dict()
        frame_pred_set = set()
        dialogue_act_pred_set = set()
        intent_pred_set = set()
        frame_gold_set = set()
        dialogue_act_gold_set = set()
        intent_gold_set = set()
        entities_gold = []
        entities_pred = []
        current_frame_element_gold = ""
        current_frame_element_pred = ""
        for intent_token in example["intent_gold"]:
            intent_gold_set.add(intent_token)
        for intent_token in example["intent_pred"]:
            intent_pred_set.add(intent_token)
        for frame_element_token, token in zip(
            example["frame_element_gold"], example["tokens"]
        ):
            if frame_element_token == "O":
                continue
            if frame_element_token.startswith("B-"):
                entity_gold = dict()
                entities_gold.append(entity_gold)
                current_frame_element_gold = frame_element_token[2:]
                entity_gold[current_frame_element_gold] = [token]
            else:
                if frame_element_token[2:] == current_frame_element_gold:
                    entity_gold[current_frame_element_gold].append(token)
                else:
                    entity_gold = dict()
                    entities_gold.append(entity_gold)
                    current_frame_element_gold = frame_element_token[2:]
                    entity_gold[current_frame_element_gold] = [token]
        for frame_element_token, token in zip(
            example["frame_element_pred"], example["tokens"]
        ):
            if frame_element_token == "O":
                continue
            if frame_element_token.startswith("B-"):
                entity_pred = dict()
                entities_pred.append(entity_pred)
                current_frame_element_pred = frame_element_token[2:]
                entity_pred[current_frame_element_pred] = [token]
            else:
                if frame_element_token[2:] == current_frame_element_pred:
                    entity_pred[current_frame_element_pred].append(token)
                else:
                    entity_pred = dict()
                    entities_pred.append(entity_pred)
                    current_frame_element_pred = frame_element_token[2:]
                    entity_pred[current_frame_element_pred] = [token]

        new_example["tokens"] = example["tokens"]
        new_example["dialogue_act_gold"] = list(dialogue_act_gold_set)
        new_example["dialogue_act_pred"] = list(dialogue_act_pred_set)
        new_example["frame_gold"] = list(frame_gold_set)
        new_example["frame_pred"] = list(frame_pred_set)
        new_example["intent_gold"] = list(intent_gold_set)
        new_example["intent_pred"] = list(intent_pred_set)
        new_example["entities_gold"] = entities_gold
        new_example["entities_pred"] = entities_pred
        squeezed_predictions.append(new_example)
    return squeezed_predictions


