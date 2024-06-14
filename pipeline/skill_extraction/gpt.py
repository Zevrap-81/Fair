from dotenv import load_dotenv
from openai import OpenAI
import os, time
import numpy as np
from tqdm import tqdm

def create_prompt_cv(text):
    return f'Given the text "{text}". Extract all skills mentioned in that text. Also generate a score from 0 to 100 on how well that person can perform in a skill. 0 meaning no experience, 20 meaning beginner and 80 meaning advanced. Only give 100 if the person is an expert in the field. Give me the result as json within an array with the properties skill and score.'


def create_prompt_job_announce(text):
    return f'Given the text "{text}". Extract all skills the person needs for that job. Only include skills that are mentioned. Also generate a score from 0 to 100 on how important that skill is for this job based on the text. 0 meaning not necessary, 20 meaning optional and 80 meaning neccesary. Only give 100 if the person has to be an expert in the field. Give me the result as json within an array with the properties skill and score.'


def execute_gpt(prompt):

    # load openai key
    load_dotenv()
    OPENAI_KEY = os.getenv('OPENAI_KEY')

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=OPENAI_KEY,
    )

    messages = [{
        "role": "user",
        "content": prompt
    }]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        response_format =  {
            "type": "json_object"
        }
    )

    # return response, response_message
    try:
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return False


def extract_skills(text_arr, cv=False, job_announce=False):
    print('-- Extracting skills --')
    response_arr = np.array([])
    exception_count = 0
    with tqdm(
        # bar_format="Exceptions: {postfix} | Elapsed: {elapsed} | {rate_fmt}",
        postfix=f"Errors: {exception_count}",
        total=len(text_arr)
    ) as t:
        for text in text_arr:
            try:
                try_count = 0
                while try_count < 2:

                    if cv:
                        response = execute_gpt(create_prompt_cv(text))
                    elif job_announce:
                        response = execute_gpt(create_prompt_job_announce(text))
                    else:
                        return False
                    
                    if response:
                        response_arr = np.append(response_arr, response)
                        break
                    else:
                        # failed response try again (2 times)
                        time.sleep(10)
                        if try_count == 2:
                            exception_count += 1
                            t.postfix = exception_count
                            response_arr = np.append(response_arr, '')
                            break
                        try_count += 1
            except Exception as e:
                print(e)
                exception_count += 1
                t.postfix = exception_count
                response_arr = np.append(response_arr, '')
            t.update(1)
    
    print('-- Skills extracted --')
    return response_arr