# https://en.wikipedia.org/wiki/List_of_ICD-9_codes_001%E2%80%93139:_infectious_and_parasitic_diseases
import pickle


f = open('/Users/sr0215/Python/Clinical/Bayes/Refinement/parse.txt')

D = {}
for l in f.readlines():
    l = l.replace('\n', '')
    # print(l.split('\t'))

    [ranges, category] = l.split('\t')

    [low, high] = ranges.split('–')
    D[high] = category

print(D)
pickle.dump(D, open('/Users/sr0215/Python/Clinical/Bayes/Refinement/parse.p', 'wb'))

