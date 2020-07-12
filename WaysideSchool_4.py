def all_different(list):
    return len(list) == len(set(list))
    
for p in range(1,10):    
    for u in range(10):
        for r in range(10):
            for l in range(10):
                for e in range(10):
                    for c in range(1,10):
                        for o in range(10):
                            for m in range(1,10):
                                for t in range(10):
                                    for z in range(1,10):
                                        purple = p*100000 + u*10000 + r*1000 + p*100 + l*10 + e
                                        comet = c*10000 + o*1000 + m*100 + e*10 + t
                                        meet = m*1000 + e*100 + e*10 + t
                                        zzzzzz = z*100000 + z*10000 + z*1000 + z*100 + z*10 + z
                                        if all_different([p,u,r,l,e,c,o,m,t,z]) and purple + comet + meet == zzzzzz:
                                            print("%s + %s + %s = %s" % (purple,comet,meet,zzzzzz))