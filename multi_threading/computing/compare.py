from sentence_transformers import SentenceTransformer, models, util
import numpy as np

"""
Compute the similarity between two sentences using a given sentence similarity model.

Args:
    func_model (Model): The similarity model used to compute the embeddings.
    sentence_1 (str): The first sentence.
    sentence_2 (str): The second sentence.
    percent (bool, optional): Whether to return the similarity as a percentage. 
        If set to True, the similarity will be scaled from -1 to 1 and then converted to a percentage between 0 and 100.
        If set to False, the similarity will be returned as a floating-point value between -1 and 1.

Returns:
    float:  The similarity between the two sentences. If percent is True, the similarity will be a percentage between 0 and 100. 
            If percent is False, the similarity will be a floating-point value between -1 and 1.
"""
def get_similarity(func_model, sentence_1, sentence_2, percent=False):
    # compute embeddings
    sent1_embedding = func_model.encode(sentence_1, convert_to_tensor=True).cpu()
    sent2_embedding = func_model.encode(sentence_2, convert_to_tensor=True).cpu()
    
    # compare embeddings
    corr = util.pytorch_cos_sim(sent1_embedding, sent2_embedding)
    
    if percent:
        # return in percent (value can be from -1 to 1)
        return (corr.numpy()[0][0] + 1) / 2
    return corr.numpy()[0][0]


"""
Calculate the similarity score between two sets of skills using a given model.
    Method 2:   1. comparing highest skill pairs
                2. save pair and remove from computation list
                3. generate next highest skill pair

Parameters:
    model (object): The model used to calculate the similarity score.
    skill_eval_arr (list): An array of dictionaries representing the skills to be evaluated.
    base_arr (list): An array of dictionaries representing the base skills for comparison.
    percent (bool, optional): Whether to return the similarity score as a percentage. Defaults to True.

Returns:
    similarity_score (float): The calculated similarity score between the two sets of skills.
"""
def compare_skill_arr_method_2(model, skill_eval_arr, base_arr, percent=True):
    similarity_score = 0
    global_weight = np.array([])

    weight = np.array([])
    weight_base = np.array([])
    similarity = np.array([])
    skills_combined = []
    # generate scores of all combinations
    for cv_skill in skill_eval_arr:
        for base_skill in base_arr:
            similarity = np.append(similarity, get_similarity(model, base_skill['skill'], cv_skill['skill'], percent=percent))
            weight = np.append(weight, cv_skill['score'])
            weight_base = np.append(weight_base, base_skill['score'])
            skills_combined.append([base_skill['skill'], cv_skill['skill']])
    
    
    count = 0
    # get highest similarity
    while (len(skills_combined) > 0):
        highest_idx = similarity.argmax(axis=0)
        skills_match = skills_combined[highest_idx]

        #print(f"{skills_match[0]} with {skills_match[1]} ({weight[highest_idx]}) -> {similarity[highest_idx]}")

        # get similarity of highest skill
        tmp_weight = weight[highest_idx] * weight_base[highest_idx]
        similarity_score += similarity[highest_idx] * tmp_weight
        global_weight = np.append(global_weight, tmp_weight)
        
        # remove highest entry from data
        similarity = np.delete(similarity, highest_idx)
        weight = np.delete(weight, highest_idx)
        skills_combined.pop(highest_idx)

        # remove all entries with matched base skill
        new_similarity = []
        new_weight = []
        new_weight_base = []
        new_skills_combined = []

        for i, skills in enumerate(skills_combined):
            if not (skills[0].__eq__(skills_match[0]) or skills[1].__eq__(skills_match[1])):
                new_similarity.append(similarity[i])
                new_weight.append(weight[i])
                new_weight_base.append(weight_base[i])
                new_skills_combined.append(skills)

        similarity = np.array(new_similarity)
        weight = np.array(new_weight)
        weight_base = np.array(new_weight_base)
        skills_combined = new_skills_combined

        count += 1


    similarity_score = similarity_score / (global_weight.mean() * count )
    
    return similarity_score


"""
Calculate the similarity between two arrays of skills using a weighted average method.

Parameters:
    model: The machine learning model used to calculate the similarity between skills.
    skill_eval_arr: The array of skill evaluation data to compare.
    base_arr: The base array of skill evaluation data for comparison.
    percent: A boolean indicating whether the similarity should be calculated as a percentage. Default is True.

Returns:
    similarity: The similarity score between the two arrays of skills.
"""
def compare_skill_arr_method_1(model, skill_eval_arr, base_arr, percent=True):
    similarity = 0
    weight = np.array([])
    weight_base = np.array([])
    for data2 in skill_eval_arr:
        for data1 in base_arr:
            similarity += (get_similarity(model, data1['skill'], data2['skill'], percent=percent) * data2['score'] * data1['score'])
            weight = np.append(weight, data2['score'])
            weight_base = np.append(weight_base, data1['score'])
    similarity = similarity / (len(skill_eval_arr) * weight.mean() * weight_base.mean() * len(base_arr))
    
    return similarity