from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np


true_labels_file = "reference_output/DataSet_Misinfo_first100.corrected"
predicted_labels_file = "corrected_output/DataSet_Misinfo_first100.corrected"
categories = ["true", "false", "mixed"]


def read_labels(file_path):
    with open(file_path, "r") as file:
        labels = [line.strip() for line in file.readlines()]
    return labels


def evaluate_performance(true_labels, predicted_labels, categories):
    metrics_summary = {"precision": [], "recall": [], "fscore": []}

    for category in categories:
        true_binary = [1 if label == category else 0 for label in true_labels]
        predicted_binary = [1 if label == category else 0 for label in predicted_labels]
        tn, fp, fn, tp = confusion_matrix(true_binary, predicted_binary).ravel()

        precision, recall, fscore, _ = precision_recall_fscore_support(
            true_binary, predicted_binary, beta=0.5, average="binary", zero_division=0
        )

        # Store metrics for macro-average calculation
        metrics_summary["precision"].append(precision)
        metrics_summary["recall"].append(recall)
        metrics_summary["fscore"].append(fscore)

        print(f"=================== {category} =====================")
        print(f"TP\tFP\tFN\tPrec\tRec\tF0.5")
        print(f"{tp}\t{fp}\t{fn}\t{precision:.4f}\t{recall:.4f}\t{fscore:.4f}")
        # print("==================================")

    # Calculate and print macro-averaged metrics
    macro_precision = np.mean(metrics_summary["precision"])
    macro_recall = np.mean(metrics_summary["recall"])
    macro_fscore = np.mean(metrics_summary["fscore"])

    print("=========== Overall (Macro-average) ===========")
    print(f"Prec\tRec\tF0.5")
    print(f"{macro_precision:.4f}\t{macro_recall:.4f}\t{macro_fscore:.4f}")
    # print("===============================================")

    # Calculate and print micro-averaged metrics
    micro_precision, micro_recall, micro_fscore, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, beta=0.5, average="micro"
    )
    print("=========== Overall (Micro-average) ===========")
    print(f"Prec\tRec\tF0.5")
    print(f"{micro_precision:.4f}\t{micro_recall:.4f}\t{micro_fscore:.4f}")
    print("===============================================")


true_labels = read_labels(true_labels_file)
predicted_labels = read_labels(predicted_labels_file)

evaluate_performance(true_labels, predicted_labels, categories)
