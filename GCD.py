def gcd(a,b):
    while True:
        if a >= b:
            a = a%b
        else:
            b = b%a
        if a == 0:
            return b
        if b == 0:
            return a
            
def gcd3(a,b,c):
    return gcd(gcd(a, b), c)
    
print(gcd(36,64))
print(gcd(1071,1029))
print(gcd3(8,36,48))
print(gcd3(22,33,77))