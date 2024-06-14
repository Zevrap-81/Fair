from urllib.parse import urlencode
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

import selenium_misc.crawl as crawl

"""
Generate the URL for an Indeed job search given the keyword, location, and offset.

Args:
    keyword (str): The keyword to search for.
    location (str): The location to search in.
    offset (int, optional): The offset of entries of the search results. Defaults to 0.

Returns:
    str: The URL for the Indeed job search.
"""
def get_indeed_search_url(keyword: str, location='', offset=0):
    if location == '':
        parameters = {"q": keyword, "filter": 0, "start": offset}
    else:
        parameters = {"q": keyword, "l": location, "filter": 0, "start": offset}
    return "https://www.indeed.com/jobs?" + urlencode(parameters)


"""
Extracts details from an HTML page obtained from Indeed job listings.

Args:
	param html: The HTML content of the page.
	type html: str

Returns: A tuple containing two lists: `id_arr` and `title_arr`.
    id_arr: A list of job IDs extracted from the HTML.
	title_arr: A list of job titles extracted from the HTML.
    href_arr: A list of job links extracted from the HTML.
"""
def extract_details_from_indeed(html: str):

    soup = BeautifulSoup(html, 'lxml')

    # find parent element
    elements = soup.find_all('li', class_='css-5lfssm')

    id_arr = []
    title_arr = []
    href_arr = []

    # find all id and title (skip because of add in the middle)
    for elem in elements:
        try:
            id = elem.find(class_='jcs-JobTitle').get('data-jk')
            href = elem.find(class_='jcs-JobTitle').get('href')
            title = elem.find(class_='jcs-JobTitle').get_text()
            id_arr.append(id)
            title_arr.append(title)
            href_arr.append(href)
        except:
            continue
    
    return id_arr, title_arr, href_arr


"""
Crawls n entries for a given keyword.

Args:
    keyword (str): The keyword to search for.
    n (int, optional): The number of entries to crawl. Defaults to 10.

Returns:
    pandas.DataFrame: A DataFrame containing the crawled entries with columns:
        - title: The title of the entry.
        - id: The ID of the entry.
        - job_description: The description of the job.

Raises:
    Exception: If an error occurs during the crawling process.
"""
def crawl_n_entries(keyword: str, n=10):
    print(f"Crawling {n} entries for {keyword}:")

    job_description_arr = []
    title_arr_new = []
    id_arr_new = []
    
    offset = 0
    try:
        with tqdm(total=n) as pbar:
            while len(job_description_arr) < n:

                URL = get_indeed_search_url(keyword, offset=offset)
                
                html = crawl.get_website(URL)
                id_arr, title_arr, href_arr = extract_details_from_indeed(html)

                job_description_arr = []
                title_arr_new = []
                id_arr_new = []

                i = 0
                for id, title, href in zip(id_arr, title_arr, href_arr):
                    # exit if n is reached
                    if i > n:
                        break

                    # get details html
                    html = crawl.get_website('https://www.indeed.com' + href, wait=True, wait_selector='jobDescriptionText', cloudflare=True)
                    soup = BeautifulSoup(html, 'lxml')
                    job_description = soup.find(id='jobDescriptionText').get_text()
                    job_description_arr.append(job_description)
                    title_arr_new.append(title)
                    id_arr_new.append(id)
                    pbar.update(1)
                    i = i+1

                offset = offset + len(job_description_arr)
    except Exception as e:
        print(e)

    return pd.DataFrame({'title': title_arr_new, 'id': id_arr_new, 'category': keyword, 'job_description': job_description_arr})
    
