end = 400

cacheA = [a*a-1 for a in range(end)]
cacheB = [b*b*b for b in range(end)]

def search() :
    for a in range(1,end):
        for b in range(a,end):
            for c in range(b,end):
                for d in range(c,end):
                    if cacheB[a] + cacheB[b] + cacheB[c] + cacheB[d] == cacheA[a]*b*c*d + 2019:
                        print([a,b,c,d])
                        return
search()
print("done")