def relativePrime(m, n) :
    for i in range(2,m+1):
        if m%i==0 and n%i==0:
            return False
    return True
    
#print relativePrime(2,4)
#print relativePrime(3,5)

rel15 = [x for x in range(1,200) if relativePrime(15, x)]
print(rel15, len(rel15))

rel24 = [x for x in range(1,200) if relativePrime(24, x)]
print(rel24, len(rel24))

rel15or24 = [x for x in range(1,200) if relativePrime(15,x) or relativePrime(24,x)]
print(rel15or24, len(rel15or24))