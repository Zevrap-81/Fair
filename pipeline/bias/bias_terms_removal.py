import re
import pandas as pd
import numpy as np
from transformers import pipeline
import validators
from tqdm import tqdm

def replace_unusable_characters(text):
    # remove all characters besides numbers, letters and ".?!,;:()[]"
    cleaned_text = re.sub(r'[^\w\s.?!,;:()@+\/\\[\]]', ' ', text)

    # replace multiple whitespaces with single whitespace & strip
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def load_ner_model(model_name="dslim/bert-large-NER"):
    ner_classifier = pipeline("ner", model=model_name, aggregation_strategy="first")
    return ner_classifier


def extractEmails(text, replace=True):
    # detects all e-mails but also text like email@...com
    #pattern = r'(?:[a-z0-9+!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])'
    
    # detects all e-mails but also text like email@...com
    pattern = r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)'
    
    emails = re.findall(pattern, text, re.IGNORECASE)
    if replace:
        cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    else:
        cleaned_text = text
    
    return emails, cleaned_text


def extractLinks(text, replace=True):
    # regex to detect all links in a text
    # pattern = r'(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})(\.[a-zA-Z0-9]{2,})?'
    pattern = r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+'
    
    links = re.findall(pattern, text, re.IGNORECASE)

    val_links = []
    cleaned_text = text
    
    for link in links:
        if not link.startswith('http'):
            tmp_link = "https://" + link
        else:
            tmp_link = link
        
        if validators.url(tmp_link):
            val_links.append(link)

            if replace:
                cleaned_text = re.sub(re.escape(link), '', cleaned_text, 1)
        else:
            pass
            # print(link)
    
    return val_links, cleaned_text


def extract_loc_and_per(ner_model, text, proc_text, replace=True):
    location = []
    person = []
    
    classified = ner_model(text)
    
    for entry in classified:
        if entry["entity_group"] == 'LOC':
            location.append(entry)
        elif entry["entity_group"] == 'PER':
            person.append(entry)
            
    # replace entries in text
    if replace:
        for entry in location:
            pattern = r'\b{}\b'.format(re.escape(entry['word']))
            proc_text = re.sub(pattern, "", proc_text)
        for entry in person:
            pattern = r'\b{}\b'.format(re.escape(entry['word']))
            proc_text = re.sub(pattern, "", proc_text)
        
    return proc_text, person, location


def remove_stopwords(stopwords, text):

    extracted_words = []

    for word in stopwords:
        # Search for the word in the string
        match = re.search(r'\b' + re.escape(word) + r'\b', text)
        if match:
            # Extract the word
            extracted_words.append(match.group(0))

            # Remove the word from the string
            text = re.sub(r'\b' + re.escape(word) + r'\b', '', text)
    
    return text, extracted_words


def remove_bias_pipeline(text_arr, label_arr, ner_model) -> pd.DataFrame:

    print('-- Removing bias terms --')

    df = pd.DataFrame(columns=['label', 'orig_text', 'processed_text', 'email', 'person', 'location', 'linkedin_xing'])

    orig_text_arr = pd.Series(text_arr, dtype=str)
    cleaned_text_arr = pd.Series(dtype=str)
    emails_arr = pd.Series(dtype=str)
    links_arr = pd.Series(dtype=str)
    person_arr = pd.Series(dtype=str)
    location_arr = pd.Series(dtype=str)
    pronouns_arr = pd.Series(dtype=str)
    gender_terms_arr = pd.Series(dtype=str)


    with tqdm(total=len(text_arr)) as pbar:
        for i in range(len(text_arr)):

            # remove emails
            emails, cleaned_text = extractEmails(orig_text_arr.iloc[i])
            emails_arr = pd.concat([emails_arr, pd.Series([emails])], ignore_index=True)
            cleaned_text_arr = pd.concat([cleaned_text_arr, pd.Series([cleaned_text])], ignore_index=True)

            # remove links
            links, cleaned_text = extractLinks(cleaned_text_arr.iloc[i])
            cleaned_text_arr.iloc[i] = cleaned_text
            links_arr = pd.concat([links_arr, pd.Series([links])], ignore_index=True)

            # remove name and location
            cleaned_text, person, location = extract_loc_and_per(ner_model, orig_text_arr.iloc[i], cleaned_text_arr.iloc[i])
            cleaned_text_arr.iloc[i] = cleaned_text
            person_arr = pd.concat([person_arr, pd.Series([person])], ignore_index=True)
            location_arr = pd.concat([location_arr, pd.Series([location])], ignore_index=True)

            # sexuality terms
            gender_terms = ['male', 'female', 'cisgender', 'asexual', 'aromantic', 'intersex', 'nonbinary',
                            'transgender', 'bisexual', 'gay', 'lesbian', 'queer', 'pansexual']

            pronouns = ['she', 'her', 'hers', 'they', 'them', 'their', 'he', 'him', 'his',
                        'fae', 'faer', 'faers', 'ze', 'hir', 'hirs']
            
            proc_text, extracted = remove_stopwords(gender_terms, cleaned_text_arr.iloc[i])
            gender_terms_arr = pd.concat([gender_terms_arr, pd.Series([extracted])], ignore_index=True)
            proc_text, extracted = remove_stopwords(pronouns, proc_text)
            pronouns_arr = pd.concat([pronouns_arr, pd.Series([extracted])], ignore_index=True)
            cleaned_text_arr.iloc[i] = proc_text


            pbar.update(1)


    print('Merging dataframes...')
    df['label'] = label_arr
    df['processed_text'] = cleaned_text_arr
    df['email'] = emails_arr
    df['linkedin_xing'] = links_arr
    df['person'] = person_arr
    df['location'] = location_arr
    df['pronouns'] = pronouns_arr
    df['gender_terms'] = gender_terms_arr

    print('-- done --')
    return df