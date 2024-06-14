import os, glob

def get_files_and_values(path):
    files = [os.path.normpath(f) for f in glob.glob(os.path.join(path, "*.csv"))]
    file_values = []
    for file in files:
        level = ''
        model = ''

        tmp_arr = os.path.basename(file).split('_')
        model = '_'.join(tmp_arr[:2]).split('.')[0]

        if len(tmp_arr) > 2:
            level = '_'.join(tmp_arr[2:]).split('.')[0]
        else:
            level = ''

        file_values.append((file, model, level))

    return file_values