import re, os


def replace_unusable_characters(text):
    # remove all characters besides numbers, letters and ".?!,;:()[]"
    cleaned_text = re.sub(r'[^\w\s.?!,;:()@+\/\\[\]]', ' ', text)

    # replace multiple whitespaces with single whitespace & strip
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def extract_text(file):
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def replace_new_line(text):
    return text.replace(r"\r\n", ' ')


def get_all_files(directory_path, file_type):
    file_list = []
    for foldername, subfolders, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith(file_type.lower()):
                file_list.append(os.path.join(foldername, filename))
    return file_list