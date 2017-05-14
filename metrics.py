from nltk.corpus import wordnet as wn

word_pair_dict = {}

def dirac_measure(a, b):
    """
    Returns 1 iff a = b and 0 otherwise.
    """
    return float(a == b)


def wup_measure(a, b, similarity_threshold = 0.925, debug = False):
    """
    Returns Wu-Palmer similarity score.
    More specifically, it computes:
        max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
        where interp is a 'interpretation field'
    """
    if debug: print 'Original', a, b
    if word_pair_dict.has_key(a+','+b):
        return  word_pair_dict[a+','+b]

    def get_semantic_field(a):
        return wn.synsets(a, pos=wn.NOUN)

    if a == b: return 1.0

    interp_a = get_semantic_field(a) 
    interp_b = get_semantic_field(b)
    if debug: print(interp_a)

    if interp_a == [] or interp_b == []:
        return 0.0

    if debug: print 'Stem', a, b
    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y)
            if debug: print 'Local', local_score
            if local_score > global_max:
                global_max=local_score
    if debug: print 'Global', global_max

    # we need to use the semantic fields and therefore we downweight
    # unless the score is high which indicates both are synonyms
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score = global_max * interp_weight
    word_pair_dict[a+','+b] = final_score
    return final_score 


def compute_wups(input_gt, input_pred, thresh):
    global word_pair_dict
    word_pair_dict = {}
    if thresh == -1:
        measure = dirac_measure
    else:
        measure = lambda x, y: wup_measure(x, y, thresh)

    # if thresh == -1:
    #     print 'standard Accuracy is used'
    # else:
    #     print 'soft WUPS is used'
    score_list = [measure(ta, pa) for (ta, pa) in zip(input_gt, input_pred)]
    final_score = sum(map(
        lambda x: float(x) / float(len(score_list)), score_list))

    # print 'final score:', final_score
    return final_score
