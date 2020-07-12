import itertools

def assignvalue(l):
    for permutation in l:
        for val in range(len(l)):
            digits = list(itertools.combinations('1234567890',len(l)))
            permutation[val] = digits[val]
            print(digits)
    return l


assignvalue(['a', 'b', 'c'])
#print(assignvalue(['a', 'b', 'c']))

def tupleToNumber(t):
    sum = 0
    for n in t:
        sum = sum * 10 + n
    return sum

print(tupleToNumber((1,2,3)))
