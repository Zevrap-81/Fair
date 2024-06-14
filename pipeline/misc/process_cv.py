import os
import pandas as pd
from misc.process_text import get_all_files, extract_text, replace_unusable_characters

def load_cvs_or_job_announce(path, replace=True, folder=False):
    cv_df_tmp = pd.DataFrame(
         columns=['label', 'orig_text', 'processed_text', 'email', 'person', 'location', 'linkedin_xing']
    )
    
    if folder:
        for name in os.listdir(path):
            print(f"Looping through folder: {name}")
            for file in get_all_files(os.path.join(path, name), '.txt'):
                text = extract_text(file)
                if replace:
                    text = replace_unusable_characters(text)
                cv_df_tmp = pd.concat([
                    cv_df_tmp,
                    pd.DataFrame([{
                          'orig_text': text,
                          'label': os.path.basename(file).split('.')[0]
                    }])
                ], ignore_index=True)
    else:
        for file in get_all_files(path, '.txt'):
            text = extract_text(file)
            if replace:
                    text = replace_unusable_characters(text)
            cv_df_tmp = pd.concat([
                cv_df_tmp,
                pd.DataFrame([{
                    'orig_text': text,
                    'label': os.path.basename(file).split('.')[0]
                }])
            ], ignore_index=True)
    
    return cv_df_tmp