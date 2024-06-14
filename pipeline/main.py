import pandas as pd
import torch

import skill_extraction.gpt as gpt
from misc.process_cv import load_cvs_or_job_announce
import bias.bias_terms_removal as btr


def main():
    # df = load_cvs_or_job_announce(f'./resume/FOLDER/')
    # df = pd.read_csv('./data/ResumeDataSet_processed.csv', index_col=0)
    df = pd.read_csv('./data/job_announce_result_10.csv', index_col=0)

    # remove bias terms
    ner_model = btr.load_ner_model()
    df_bias = btr.remove_bias_pipeline(df['job_description'].values, df['category'].values, ner_model)

    # remove ner model
    del ner_model
    torch.cuda.empty_cache()

    # extract and evaluate skills
    response = gpt.extract_skills(df_bias['processed_text'].values, cv=False, job_announce=True)
    df_bias['skill_eval_gpt'] = response

    df_bias.to_csv('./data/job_announce_result_final.csv')

    # add job title (for job announce)
    df_bias['job_title'] = df['title']

    df_bias.to_csv('./data/job_announce_result_final.csv')



if __name__ == '__main__':
    main()