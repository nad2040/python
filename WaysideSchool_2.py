def all_different(list):
    return len(list) == len(set(list))

for s in range(10):
    for e in range(10):
        for v in range(10):
            for n in range(10):
                for f in range(10):
                    for o in range(10):
                        for r in range(10):
                            for t in range(10):
                                for y in range(10):
                                    seven = s * 10000 + e * 1000 + v * 100 + e * 10 + n
                                    forty9 = f * 100000 + o * 10000 + r * 1000 + t * 100 + y * 10 + 9
                                    if all_different([s,e,v,n,f,o,r,t,y,9]) and seven * 7 == forty9:
                                        print("%s * %d = %s" % (seven, 7, forty9))