import contextlib
import joblib
from itertools import product
from tqdm import tqdm
import pandas as pd
from ast import literal_eval
from sentence_transformers import SentenceTransformer

from computing.compare import compare_skill_arr_method_2, compare_skill_arr_method_1


"""
Context manager to patch joblib to report into tqdm progress bar given as argument
    
Parameters:
    tqdm_object (tqdm.tqdm): The tqdm progress bar object to report the progress into.
    
Returns:
    tqdm.tqdm: The same tqdm progress bar object to be used within the context.
"""


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


"""
Calculate the similarity between a single CV and a single job using a given SentenceTransformer model.

Parameters:
    model (SentenceTransformer): The SentenceTransformer model used for calculating the similarity.
    cv_row (pd.Series): The row representing the CV.
    job_row (pd.Series): The row representing the job.
    percent (bool, optional): Determines whether the similarity should be returned as a percentage. Defaults to True.

Returns:
    dict: A dictionary containing the following keys:
        "cv_index":     The index of the CV.
        "job_index":    The index of the job.
        "cv_label":     The label of the CV.
        "job_label":    The label of the job.
        "similarity":   The similarity score between the CV and the job.
"""


def calculate_single_similarity(
    model: SentenceTransformer,
    cv_row: pd.Series,
    job_row: pd.Series,
    method: int,
    percent=True,
):
    try:
        if method == 1:
            similarity = compare_skill_arr_method_1(
                model,
                literal_eval(cv_row["skills"]),
                literal_eval(job_row["skills"]),
                percent=percent,
            )
        else:
            similarity = compare_skill_arr_method_2(
                model,
                literal_eval(cv_row["skills"]),
                literal_eval(job_row["skills"]),
                percent=percent,
            )

        return {
            "cv_index": cv_row["file_label"],
            "job_index": job_row["job_label"],
            "cv_label": cv_row["job_label"],
            "job_label": job_row["job_label"],
            "similarity": similarity,
        }
    except Exception as e:
        print(e)
        return {
            "cv_index": cv_row["file_label"],
            "job_index": job_row["job_label"],
            "cv_label": cv_row["job_label"],
            "job_label": job_row["job_label"],
            "similarity": 0,
        }


"""
Calculates the similarity between each pair of CV and job descriptions in parallel.

Parameters:
    model (SentenceTransformer): The SentenceTransformer model used for calculating similarity.
    cv_result (pd.DataFrame): The DataFrame containing CV descriptions.
    job_result (pd.DataFrame): The DataFrame containing job descriptions.
    n_jobs (int, optional): The number of parallel jobs to run. Defaults to 1 (multithreading disabled).
    description (str, optional): A description of the job task. Defaults to an empty string.
    percent (bool, optional): Determines whether the similarity should be returned as a percentage. Defaults to True.

Returns:
    pd.DataFrame: A DataFrame containing the similarity scores between each pair of CV and job descriptions.
"""


def calculate_job_announce_parallel(
    model: SentenceTransformer,
    cv_result: pd.DataFrame,
    job_result: pd.DataFrame,
    n_jobs=1,
    method=2,
    description="",
    percent=True,
):
    results = []
    num_tasks = len(cv_result) * len(job_result)

    # print('tasks with length: ', num_tasks)

    # Integrate tqdm for progress visualization
    with tqdm_joblib(tqdm(desc=description, total=num_tasks)) as pbar:
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(calculate_single_similarity)(
                model, cv_row, job_row, method, percent=percent
            )
            for (_, cv_row), (_, job_row) in product(
                cv_result.iterrows(), job_result.iterrows()
            )
        )

    return pd.DataFrame(results)
