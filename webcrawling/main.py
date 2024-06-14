import pandas as pd

import selenium_misc.crawl as crawl
import web_processing.indeed as indeed


def main():

    # load data set for categories
    df = pd.read_csv('./data/ResumeDataSet_processed.csv', index_col=0)

    # crawl job announces for each category
    df_result_full = pd.DataFrame(columns=['title', 'category',  'id', 'job_description'])
    for category in df['Category'].unique():
        df_result = indeed.crawl_n_entries(category, n=15)
        df_result_full = pd.concat([df_result_full, df_result], ignore_index=True)

        # presave each category
        df_result.to_csv(f'./tmp/job_announce_{category}.csv')

    # save job announces to file
    print('saving to file...')
    df_result_full.to_csv('./data/job_announce_result.csv')


if __name__ == '__main__':
    main()