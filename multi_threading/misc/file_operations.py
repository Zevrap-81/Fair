import os, string, random
import pandas as pd


"""
Generates a random string of the specified length.

Parameters:
    length (int): The length of the random string to generate. Defaults to 15.

Returns:
    str: The randomly generated string.
"""
def generate_random_string(length=15):
    characters = string.ascii_letters + string.digits  # Combining letters and digits
    return ''.join(random.choice(characters) for _ in range(length))


"""
Save a dataframe as a CSV file. If the path does not exist, it will be created. Emergency save if exceptions occurs.

Parameters:
    path (str): The path where the file will be saved.
    filename (str): The name of the file.
    data (pandas.DataFrame): The dataframe to be saved.
    index (bool, optional): Whether to include the index in the CSV file. Default is True.

Returns:
    None
"""
def save_dataframe(path, filename, data, index=True):
    if not os.path.exists(path):
        os.makedirs(path)
    try:
        data.to_csv(os.path.join(path, filename), index=index)
        print(f'saved file in: {os.path.join(path, filename)}.csv')
    except Exception as e:
        print(e)
        print('-- EMERGENCY SAVE --')
        # generate random string of length 15
        random_string = generate_random_string(length=15)
        data.to_csv(os.path.join('./', random_string, '.csv'), index=index)
        print(f'saved file as: {random_string}.csv')
        print('--------------------')
