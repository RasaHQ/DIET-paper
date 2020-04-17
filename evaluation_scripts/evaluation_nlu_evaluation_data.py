import converters
import argparse
import os
from pathlib import Path
import numpy as np


# Code taken from
# https://gitlab.com/hwu-ilab/hermit-nlu/blob/master/data/nlu_benchmark/
# to ensure same evaluation metrics as https://arxiv.org/abs/1910.00912
# Additional methods were added to be able to call desired evaluation methods.


def evaluate(predictions_file):
    json_prediction = converters.load_json_prediction_file(
        predictions_file=predictions_file
    )
    squeezed_predictions = converters.squeeze_prediction_span(
        json_prediction=json_prediction
    )

    # Initialising error variables
    intent_tp = 0.0
    intent_fp = 0.0
    intent_fn = 0.0

    entity_tp = 0.0
    entity_fp = 0.0
    entity_fn = 0.0

    combined_tp = 0.0
    combined_fp = 0.0
    combined_fn = 0.0

    for example in squeezed_predictions:
        # Intent confusion matrix
        for intent_gold in example["intent_gold"]:
            if intent_gold in example["intent_pred"]:
                intent_tp += 1
            else:
                intent_fn += 1
        for intent_pred in example["intent_pred"]:
            if intent_pred not in example["intent_gold"]:
                intent_fp += 1

        # Entity confusion matrix
        for entity_gold_temp in example["entities_gold"]:
            found = False
            for entity_gold in entity_gold_temp:
                for entity_pred_temp in example["entities_pred"]:
                    if entity_gold in entity_pred_temp:
                        found = not set(entity_gold_temp[entity_gold]).isdisjoint(
                            set(entity_pred_temp[entity_gold])
                        )
                        if found:
                            break
            if found:
                entity_tp += 1
            else:
                entity_fp += 1

        for entity_pred_temp in example["entities_pred"]:
            found = False
            for entity_pred in entity_pred_temp:
                for entity_gold_temp in example["entities_gold"]:
                    if entity_pred in entity_gold_temp:
                        found = not set(entity_pred_temp[entity_pred]).isdisjoint(
                            set(entity_gold_temp[entity_pred])
                        )
                        if found:
                            break
            if not found:
                entity_fn += 1

        combined_tp = intent_tp + entity_tp
        combined_fn = intent_fn + entity_fn
        combined_fp = intent_fp + entity_fp

    print("")

    intent_precision = (
        intent_tp / (intent_tp + intent_fp) if (intent_tp + intent_fp) > 0.0 else 0.0
    )
    intent_recall = (
        intent_tp / (intent_tp + intent_fn) if (intent_tp + intent_fn) > 0.0 else 0.0
    )
    intent_f1 = (
        (2 * intent_precision * intent_recall) / (intent_precision + intent_recall)
        if (intent_precision + intent_recall) > 0.0
        else 0.0
    )

    entity_precision = (
        entity_tp / (entity_tp + entity_fp) if (entity_tp + entity_fp) > 0.0 else 0.0
    )
    entity_recall = (
        entity_tp / (entity_tp + entity_fn) if (entity_tp + entity_fn) > 0.0 else 0.0
    )
    entity_f1 = (
        (2 * entity_precision * entity_recall) / (entity_precision + entity_recall)
        if (entity_precision + entity_recall) > 0.0
        else 0.0
    )

    combined_precision = combined_tp / (combined_tp + combined_fp)
    combined_recall = combined_tp / (combined_tp + combined_fn)
    combined_f1 = (2 * combined_precision * combined_recall) / (
        combined_precision + combined_recall
    )

    print(
        "Entity scores: P: {}, R: {}, F1: {}".format(
            entity_precision, entity_recall, entity_f1
        )
    )

    print(
        "Intent scores: P: {}, R: {}, F1: {}".format(
            intent_precision, intent_recall, intent_f1
        )
    )

    print(
        "Combined scores: P: {}, R: {}, F1: {}".format(
            combined_precision, combined_recall, combined_f1
        )
    )

    # open and read the file after the appending:
    directory = os.path.dirname(predictions_file)
    f = open(os.path.join(directory, "hermit_results.txt"), "w")
    f.write(
        "Entity scores: P: {}, R: {}, F1: {}".format(
            entity_precision, entity_recall, entity_f1
        )
    )
    f.write("\n")
    f.write(
        "Intent scores: P: {}, R: {}, F1: {}".format(
            intent_precision, intent_recall, intent_f1
        )
    )
    f.write("\n")
    f.write(
        "Combined scores: P: {}, R: {}, F1: {}".format(
            combined_precision, combined_recall, combined_f1
        )
    )
    f.close()


def read_results(file_name):
    f = open(file_name, "r")
    lines = f.readlines()

    entity_precision = float(lines[0].split(" ")[3][:-1])
    entity_recall = float(lines[0].split(" ")[5][:-1])
    entity_fscore = float(lines[0].split(" ")[7][:-1])

    intent_score = float(lines[1].split(" ")[3][:-1])

    return entity_precision, entity_recall, entity_fscore, intent_score


def get_numbers(scores):
    avg_score = sum(scores) / len(scores)
    std = np.std(np.array(scores), axis=0, dtype=np.float64)

    avg_score = round(avg_score * 100, 2)
    std = round(float(std) * 100, 2)

    return avg_score, std


def run(folder_name):
    e_precision = []
    e_recall = []
    e_fscore = []
    i_acc = []

    for i in range (1, 11):
        folder = Path(folder_name.format(i))
        results_file = folder / "results" / "hermit_results.txt"

        try:
            entity_precision, entity_recall, entity_fscore, intent_score = read_results(results_file)
        except FileNotFoundError:
            print(f"Result file '{results_file}' is missing.")
            continue

        e_precision.append(entity_precision)
        e_recall.append(entity_recall)
        e_fscore.append(entity_fscore)
        i_acc.append(intent_score)

    if len(e_precision) != 10:
        print("STOP! Missing some values.")
        #return

    e_precision_avg, e_precision_deviation = get_numbers(e_precision)
    e_recall_avg, e_recall_deviation = get_numbers(e_recall)
    e_fscore_avg, e_fscore_deviation = get_numbers(e_fscore)
    i_acc_avg, i_acc_deviation = get_numbers(i_acc)

    print("Results for complete NLU Evaluation Dataset:")
    print("Entity:")
    print(f"P: {e_precision_avg} +-{e_precision_deviation}")
    print(f"R: {e_recall_avg} +-{e_recall_deviation}")
    print(f"F: {e_fscore_avg} +-{e_fscore_deviation}")
    print("Intent:")
    print(f"P: {i_acc_avg} +-{i_acc_deviation}")
    print(f"R: {i_acc_avg} +-{i_acc_deviation}")
    print(f"F: {i_acc_avg} +-{i_acc_deviation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLU Benchmark evaluation script")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Input file (to evaluate a single file)",
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        default=None,
        help="Folder name template (to average results of different folds)",
    )

    args = parser.parse_args()
    if args.folder is not None:
        run(args.folder)
    else:
        evaluate(args.input)
