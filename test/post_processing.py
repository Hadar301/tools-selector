import json

import numpy as np

res_file1_path = "results/normal_agent_results.json"
res_file2_path = "results/filtering_agent_results.json"


def calculate_mean_accuracy(file_path: str) -> np.float32:
    """Read a results JSON file and return the mean accuracy score."""
    with open(file_path, "r") as f:
        results = json.load(f)

    accuracies = [r["accuracy"] for r in results]
    return np.mean(accuracies)


def calculate_test_pass_ratio(file_path: str) -> np.float32:
    with open(file_path, "r") as f:
        results = json.load(f)

    test_pass = [r["test_pass"] for r in results]
    return np.sum(test_pass) / len(test_pass)


def calculate_accruacy_above_thresh_ratio(file_path: str, threshold: float) -> np.float32:
    with open(file_path, "r") as f:
        results = json.load(f)

    accuracies = np.array([r["accuracy"] for r in results])
    above_accoracies = np.where(accuracies > threshold, 1, 0)

    return np.sum(above_accoracies) / len(above_accoracies)


if __name__ == "__main__":
    normal_acc = calculate_mean_accuracy(res_file1_path)
    filtering_acc = calculate_mean_accuracy(res_file2_path)

    normal_tst_pass_ratio = calculate_test_pass_ratio(res_file1_path)
    filtering_tst_pass_ratio = calculate_test_pass_ratio(res_file2_path)

    threshold = 0.5
    normal_thresh_acc_ratio = calculate_accruacy_above_thresh_ratio(res_file1_path, threshold)
    filtering_thresh_acc_ratio = calculate_accruacy_above_thresh_ratio(res_file2_path, threshold)    

    print(f"Normal Agent Mean Accuracy:    {normal_acc*100:.4f}")
    print(f"Filtering Agent Mean Accuracy: {filtering_acc*100:.4f}")

    print(f"Normal Agent test pass to all tests ratio: {normal_tst_pass_ratio*100:.4f}")
    print(f"Filtering Agent test pass to all tests ratio: {filtering_tst_pass_ratio*100:.4f}")

    print(f"Normal Agent above threshold accuracy ratio:    {normal_thresh_acc_ratio*100:.4f}")
    print(f"Filtering Agent above threshold accuracy ratio: {filtering_thresh_acc_ratio*100:.4f}")