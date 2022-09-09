#print([(m, n) for m in range(2, 1001) for n in range(2,m/2) if ((n**m+1) % (m**n+1) == 0)])

from itertools import combinations

def factorPairs(num):
    pairs=[]
    end = int(num**0.5)+1
    for i in range(1, end):
        if num%i==0:
            pairs.append((i, num/i))
    return pairs

def goodPair(pair):
    dset1 = set(str(pair[0]))
    dset2 = set(str(pair[1]))
    dinter = dset1.intersection(dset2)
    dunion = dset1.union(dset2)
    return len(dinter)==0 and not('0' in dunion)

def pairToDset(pair):
    dset1 = set(str(pair[0]))
    dset2 = set(str(pair[1]))
    return dset1.union(dset2)

def hasTwoGoodPairs(pairs):
    comb = combinations(pairs, 2)
    for p in list(comb):
        if goodPair(p[0]) and goodPair(p[1]):
            dset1 = pairToDset(p[0])
            dset2 = pairToDset(p[1])
            if len(dset1.union(dset2))==9 and len(dset1.intersection(dset2))==0:
                print(p, p[0][0]*p[0][1])

for i in range(2000,9000):
    hasTwoGoodPairs(factorPairs(i))


