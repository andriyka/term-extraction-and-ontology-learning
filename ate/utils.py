def precision(test_terms,true_terms):
    num = float(len(set(test_terms).intersection(true_terms)))
    denum = float(len(test_terms))
    return num/denum

def recall(test_terms,true_terms):
    num = float(len(set(test_terms).intersection(true_terms)))
    denum = float(len(true_terms))
    return num / denum

def f_measure(test_terms,true_terms):
    p = precision(test_terms,true_terms)
    r = recall(test_terms,true_terms)
    return 2*p*r/(r+p)