def pythagorean_triple(n,plist):
    for a in range(1,n):
        for b in range(a,n):
            for c in range(b,n): 
                if a*a + b*b == c*c:
                    plist.append((a,b,c))
    return plist
    
print(pythagorean_triple(100,[]))

n=100
print([(a,b,c) for a in range(1,n) for b in range(a,n) for c in range(b,n) if a*a+b*b==c*c])
