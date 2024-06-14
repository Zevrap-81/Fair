**analyze:**

-   code to analyze the models
    -   (1) file_value = operations.get_files_and_values(<PATH_TO_PREDICTIONS>)
    -   (2) metrics.accuracy_evaluation(file_value, 'method1', th_start=0.5, th_end=0.95, th_step=0.05) -> accuracy evalution from until a certain threshold
    -   (3) metrics.confusion_matrix_generate(file_value_arr1, 'method1', './data/analyze/method1/', 'cv_label', 'job_label', ['operations', 'tech'], th_start=0.5, th_end=0.95, th_step=0.05) -> generation of confusion matrices
        -   ['operations', 'tech'] are excluded labels due to incosistent data -> normaly not necessary

**deberta_test:**

-   test with deberta model -> not succesfull (results in 'new_data' can be analyzed with 'analyze')

**multi_threading:**

-   computing of all the data for the confusion matrices
-   takes long therefore multithreading recommened -> ressource intensive (on normal PC only 3x multithreading recommended)

**pipline:**

-   implementation of the entire pipeline with normal code and not notebooks
-   OpenAI API Key has to be added in .env
-   pipeline can either be loaded with resumes or job annoucements the process is the same

**webcrawling:**

-   semi automated method of crawling steptone to gather the job annoucements (more information in README in folder)
