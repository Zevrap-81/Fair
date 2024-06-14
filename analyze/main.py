import misc.operations as operations
import metrics.metrics as metrics


def main():
    print("Evaluating files...")
    # file_value_arr1 = operations.get_files_and_values("./data/result_df/method1")
    # file_value_arr2 = operations.get_files_and_values("./data/result_df/method2")
    # file_value_arr3 = operations.get_files_and_values(
    #     "./data/result_df/new_data_method1"
    # )
    # file_value_arr4 = operations.get_files_and_values(
    #     "./data/result_df/new_data_method2"
    # )
    file_value_arr5 = operations.get_files_and_values(
        "./data/result_df/deberta_method2"
    )

    # print("Generating metrics...")
    # metrics_df1 = metrics.accuracy_evaluation(file_value_arr1, 'method1', th_start=0.5, th_end=0.95, th_step=0.05)
    # metrics_df2 = metrics.accuracy_evaluation(file_value_arr2, 'method2', th_start=0.5, th_end=0.95, th_step=0.05)
    # metrics_df3 = metrics.accuracy_evaluation(
    #     file_value_arr3, "method1", th_start=0.5, th_end=0.95, th_step=0.05
    # )
    # metrics_df4 = metrics.accuracy_evaluation(
    #     file_value_arr4, "method2", th_start=0.5, th_end=0.95, th_step=0.05
    # )
    metrics_df5 = metrics.accuracy_evaluation(
        file_value_arr5, "method2", th_start=0.5, th_end=0.95, th_step=0.05
    )

    # print("Saving to file...")
    # metrics_df1.to_csv('./data/analyze/method1/accuracy.csv')
    # metrics_df2.to_csv('./data/analyze/method2/accuracy.csv')
    # metrics_df3.to_csv("./data/analyze/new_data_method1/accuracy.csv")
    # metrics_df4.to_csv("./data/analyze/new_data_method2/accuracy.csv")
    metrics_df5.to_csv("./data/analyze/deberta_method2/accuracy.csv")

    print("Generating confusion matrices...")
    # metrics.confusion_matrix_generate(file_value_arr1, 'method1', './data/analyze/method1/', 'cv_label', 'job_label', ['operations', 'tech'], th_start=0.5, th_end=0.95, th_step=0.05)
    # metrics.confusion_matrix_generate(file_value_arr2, 'method2', './data/analyze/method2/', 'cv_label', 'job_label', ['operations', 'tech'], th_start=0.5, th_end=0.95, th_step=0.05)
    # metrics.confusion_matrix_generate(
    #     file_value_arr3,
    #     "method1",
    #     "./data/analyze/new_data_method1/",
    #     "cv_label",
    #     "job_label",
    #     ["operations", "tech"],
    #     th_start=0.5,
    #     th_end=0.95,
    #     th_step=0.05,
    # )
    # metrics.confusion_matrix_generate(
    #     file_value_arr4,
    #     "method2",
    #     "./data/analyze/new_data_method2/",
    #     "cv_label",
    #     "job_label",
    #     ["operations", "tech"],
    #     th_start=0.5,
    #     th_end=0.95,
    #     th_step=0.05,
    # )
    metrics.confusion_matrix_generate(
        file_value_arr5,
        "method2",
        "./data/analyze/deberta_method2/",
        "cv_label",
        "job_label",
        ["operations", "tech"],
        th_start=0.5,
        th_end=0.95,
        th_step=0.05,
    )


if __name__ == "__main__":
    main()
