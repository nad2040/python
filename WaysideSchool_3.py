def all_different(list):
    return len(list) == len(set(list))
    
for p in range(1,10):    
    for w in range(1,10):
        for h in range(10):
            for i in range(10):
                for t in range(10):
                    for e in range(10):
                        for a in range(10):
                            for r in range(10):
                                for c in range(10):
                                    for n in range(10):
                                        white = w*10000 + h*1000 + i*100 + t*10 + e
                                        water = w*10000 + a*1000 + t*100 + e*10 + r
                                        picnic = p*100000 + i*10000 + c*1000 + n*100 + i*10 + c
                                        if all_different([w,h,i,t,e,a,r,p,c,n]) and white + water == picnic:
                                            print("%s + %s = %s" % (white,water,picnic))