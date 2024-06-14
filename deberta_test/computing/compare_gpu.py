import torch
from collections import defaultdict

def get_similarity_gpu(func_model, sentence_1, sentence_2, percent=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sent1_embedding = func_model.encode(sentence_1, convert_to_tensor=True).to(device)
    sent2_embedding = func_model.encode(sentence_2, convert_to_tensor=True).to(device)
    
    corr = torch.nn.functional.cosine_similarity(sent1_embedding, sent2_embedding)
    
    if percent:
        return (corr.item() + 1) / 2 * 100
    return corr.item()

def compare_skill_arr_method_2_gpu(model, skill_eval_arr, base_arr, percent=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    skill_similarity = defaultdict(list)
    skill_weight = defaultdict(list)

    for cv_skill in skill_eval_arr:
        for base_skill in base_arr:
            similarity = get_similarity_gpu(model, base_skill['skill'], cv_skill['skill'], percent=percent)
            skill_similarity[cv_skill['skill']].append((base_skill['skill'], similarity))
            skill_weight[cv_skill['skill']].append((base_skill['skill'], cv_skill['score'] * base_skill['score']))

    similarity_score = 0
    total_weight = 0

    while skill_similarity:
        max_similarity = -float('inf')
        max_combination = None

        for cv_skill, base_skills in skill_similarity.items():
            for base_skill, similarity in base_skills:
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_combination = (cv_skill, base_skill)

        if max_combination:
            cv_skill, base_skill = max_combination
            weight = skill_weight[cv_skill][skill_similarity[cv_skill].index((base_skill, max_similarity))]
            similarity_score += max_similarity * weight
            total_weight += weight

            del skill_similarity[cv_skill]
            for key in skill_similarity:
                skill_similarity[key] = [(b, s) for b, s in skill_similarity[key] if b != base_skill]
                skill_weight[key] = [(b, w) for b, w in skill_weight[key] if b != base_skill]

    if total_weight == 0:
        return 0

    return similarity_score / total_weight
