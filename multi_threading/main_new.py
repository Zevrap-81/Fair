import pandas as pd
from ast import literal_eval
import sys, getopt, os

from computing.model import load_model, load_base_model
from multithreading.multithreading import calculate_job_announce_parallel
from misc.file_operations import save_dataframe


def main(argv):
    opts, args = getopt.getopt(
        argv,
        "m:ms:o:os:c:mn:",
        ["modules=", "module=", "output=", "outputs=", "cpu_cores=", "method_name="],
    )
    for opt, arg in opts:
        if opt == "-h":
            print(
                "test.py -m <module path> -o <output file path> -mn <method name 1/2>"
            )
            print(
                "test.py -ms <module array> -os <output file array> -mn <method name 1/2>"
            )
            sys.exit()
        elif opt in ("-ms", "--modules"):
            module_path_arr = literal_eval(arg)
        elif opt in ("-os", "--outputs"):
            output_name_arr = literal_eval(arg)
        elif opt in ("-o", "--output"):
            output_name_arr = [arg]
        elif opt in ("-m", "--module"):
            module_path_arr = [arg]
        elif opt in ("-c", "--cpu_cores"):
            cpu_cores = arg
        elif opt in ("-mn", "--method_name"):
            method = int(arg)

    # check if module path and output name are provided and of same length
    if len(module_path_arr) != len(output_name_arr):
        print("Please provide same amount of module paths and corresponding output")
        return
    elif len(module_path_arr) == 0:
        print(
            "Please provide module paths and corresponding output as lists of strings"
        )
        return
    if (method != 1) and (method != 2):
        print("Please provide method name as 1 or 2")
        return
    num_tasks = len(module_path_arr)

    # set cpu to 1 if not specified
    if cpu_cores is None:
        print("cpu cores not specified, default to 1 (mulithreading disabled)")
        cpu_cores = 1

    # load data to compare
    cv_result = pd.read_csv(
        "./data/cv_job_data/new_data/cv_result_data2.csv", index_col=0
    )
    job_result = pd.read_csv(
        "./data/cv_job_data/new_data/job_result_data2.csv", index_col=0
    )

    # start computing of similarity
    for module_path, output_name in zip(module_path_arr, output_name_arr):
        print(f"computing for: {module_path}")
        if module_path == "base_model":
            model = load_base_model()
        else:
            model = load_model(module_path, force_cuda=False)

        try:
            return_df = calculate_job_announce_parallel(
                model,
                cv_result,
                job_result,
                n_jobs=int(cpu_cores),
                method=method,
                description=os.path.basename(module_path),
                percent=True,
            )
            save_dataframe("./new_data/", output_name, return_df)
        except Exception as e:
            print(e)
            print("ERROR COMPUTING MODEL")


if __name__ == "__main__":
    main(sys.argv[1:])
