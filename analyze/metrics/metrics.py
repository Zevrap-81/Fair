import numpy as np
from tqdm import tqdm
import pandas as pd
import os.path as path

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def accuracy_f1_score(df, threshold=0.75):
    true_positive = np.array([])
    true_negative = np.array([])

    false_positive = np.array([])
    false_negative = np.array([])

    for i in range(len(df)):
        if df.iloc[i]["cv_label"] == df.iloc[i]["job_label"]:
            if df.iloc[i]["similarity"] > threshold:
                true_positive = np.append(true_positive, df.iloc[i]["similarity"])
            else:
                false_negative = np.append(false_negative, df.iloc[i]["similarity"])
        else:
            if df.iloc[i]["similarity"] > threshold:
                false_positive = np.append(false_positive, df.iloc[i]["similarity"])
            else:
                true_negative = np.append(true_negative, df.iloc[i]["similarity"])

    # accuracy with threshold
    accuracy = (len(true_positive) + len(true_negative)) / (
        len(true_positive)
        + len(true_negative)
        + len(false_positive)
        + len(false_negative)
    )

    # accuracy based on mean
    # count = (true_positive.mean() + true_negative.mean()) \
    #         / (true_positive.mean() + true_negative.mean() + false_positive.mean() + false_negative.mean())
    try:
        recall = len(true_positive) / (len(true_positive) + len(false_negative))
        precision = len(true_positive) / (len(true_positive) + len(false_positive))
        f1_score = 2 * recall * precision / (recall + precision)
    except Exception as e:
        recall = 0
        precision = 0
        f1_score = 0
        # accuracy = 0

    # print('acc: ', count)
    return accuracy, f1_score, recall, precision


def accuracy_evaluation(
    file_values: [(str, str, str)], method_str, th_start=0.5, th_end=0.95, th_step=0.05
):
    threshold_settings = np.arange(th_start, th_end, th_step)

    model_arr = []
    level_arr = []
    threshold_arr = []
    accuracy_arr = []
    f1_score_arr = []

    recall_arr = []
    precision_arr = []

    with tqdm(total=len(file_values) * len(threshold_settings)) as pbar:
        for file, model, level in file_values:
            df = pd.read_csv(file)
            for threshold in threshold_settings:
                accuracy, f1_score, recall, precision = accuracy_f1_score(df, threshold)
                model_arr.append(model)
                level_arr.append(level)
                threshold_arr.append(threshold)
                accuracy_arr.append(accuracy)
                f1_score_arr.append(f1_score)
                recall_arr.append(recall)
                precision_arr.append(precision)

                pbar.update(1)

    df_analysis = pd.DataFrame(
        {
            "model": model_arr,
            "level": level_arr,
            "threshold": threshold_arr,
            "method": method_str,
            "accuracy": accuracy_arr,
            "f1_score": f1_score_arr,
            "recall": recall_arr,
            "precision": precision_arr,
        }
    )
    return df_analysis


def confusion_matrix_generate(
    file_values: [(str, str, str)],
    method_name: str,
    save_path: str,
    cv_column_name: str,
    job_column_name: str,
    exclude_labels: list,
    th_start=0.5,
    th_end=0.95,
    th_step=0.05,
):
    threshold_settings = np.arange(th_start, th_end, th_step)

    with tqdm(total=len(file_values) * len(threshold_settings)) as pbar:
        for file, model, level in file_values:
            df = pd.read_csv(file, index_col=0)
            labels = list(set(df[cv_column_name].to_list()))

            # exclude labels from data
            df = df[~df[cv_column_name].isin(exclude_labels)]
            for label in exclude_labels:
                # df = df[df[cv_column_name] != label]
                if label in labels:
                    labels.remove(label)

            # calculate for multiple thresholds
            for threshold in threshold_settings:
                try:
                    filtered_data = df[df["similarity"] > threshold]
                    confusion = confusion_matrix(
                        filtered_data[cv_column_name], filtered_data[job_column_name]
                    )

                    plt.figure(figsize=(8, 7))
                    sns.heatmap(
                        confusion,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=labels,
                        yticklabels=labels,
                    )
                    plt.xlabel("Job Announces")
                    plt.ylabel("Resumes")
                    plt.tight_layout()

                    plt.savefig(
                        path.join(
                            save_path,
                            f"{model}_{level}_{method_name}_th{threshold:.2f}.png",
                        ),
                        dpi=300,
                    )
                    # plt.show()
                    plt.close()
                except Exception as e:
                    # print(e)
                    print(
                        f"File: {file}, model: {model}, level: {level}, threshold: {threshold}, error: {e}"
                    )

                pbar.update(1)
