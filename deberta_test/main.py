import pandas as pd
from ast import literal_eval

import sys, getopt, os

from computing.model import load_model, load_base_model, load_tokenizer
# from computing.compare import compare_skill_arr_method_2
from multithreading.multithreading import calculate_job_announce_parallel
from misc.file_operations import save_dataframe



# def calculate_job_announce(model, df, job_result):
#     df_results = pd.DataFrame(columns=['cv_index', 'job_index', 'cv_label', 'job_label', 'similarity'], dtype=float)
#     for (x, cv), (i, row) in tqdm(product(df.iterrows(), job_result.iterrows()), total=df.shape[0] * job_result.shape[0]):
#         similarity = compare_skill_arr_method_2(model, literal_eval(cv['skills']), literal_eval(row['skills']), percent=True)
#         df_results = pd.concat([
#             df_results, 
#             pd.DataFrame([{
#                 "cv_index": cv['file_label'],
#                 "job_index": row['job_label'],
#                 "cv_label": cv['job_label'],
#                 "job_label": row['job_label'],
#                 "similarity": similarity
#             }])
#         ], ignore_index=True)

#     return df_results


def main(argv):

    module_path_arr = []
    output_name_arr = []
    opts, args = getopt.getopt(argv,"m:ms:o:os:c:",["modules=","module=", "output=", "outputs=", "cpu_cores="])
    for opt, arg in opts:
        #print(opt, arg)
        if opt == '-h':
            print ('test.py -m <module path> -o <output file path>')
            print ('test.py -ms <module array> -os <output file array>')
            sys.exit()
        elif opt in ("-ms", "--modules"):
            print(arg)
            module_path_arr = literal_eval(arg)
        elif opt in ("-os", "--outputs"):
            print(arg)
            output_name_arr = literal_eval(arg)
        elif opt in ("-o", "--output"):
            output_name_arr = [arg]
        elif opt in ("-m", "--module"):
            module_path_arr = [arg]
        elif opt in ("-c", "--cpu_cores"):
            cpu_cores = arg

    # check if module path and output name are provided and of same length
    if len(module_path_arr) != len(output_name_arr):
        print(len(module_path_arr))
        print(len(output_name_arr))
        print('Please provide same amount of module paths and corresponding output')
        return
    elif len(module_path_arr) == 0:
        print('Please provide module paths and corresponding output as lists of strings')
        return
    num_tasks = len(module_path_arr)
    
    # set cpu to 1 if not specified
    if cpu_cores is None:
        print('cpu cores not specified, default to 1 (mulithreading disabled)')
        cpu_cores = 1

    # load data to compare
    cv_result = pd.read_csv('./data/new_data/cv_result_data2.csv', index_col=0)
    job_result = pd.read_csv('./data/new_data/job_result_data2.csv', index_col=0)
    
    
    # start computing of similarity
    for module_path, output_name in zip(module_path_arr, output_name_arr):
        if module_path == 'base_model':
            model = load_base_model(force_cuda=False)
            tokenizer = load_tokenizer(force_cuda=False)
            print("model and tokenizer loaded")
            print(type(model))
        else:
            model = load_model(module_path, force_cuda=False)
        return_df = calculate_job_announce_parallel(model, tokenizer, cv_result, job_result, n_jobs=int(cpu_cores), description=os.path.basename(module_path), percent=True)
        save_dataframe('./data/new_data/', output_name, return_df)
    

if __name__ == '__main__':
    main(sys.argv[1:])