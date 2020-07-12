import math

def isPrime(a):
    if a == 2 :
        return True
    if a % 2 == 0 :
        return False
    rangeEnd = int(math.sqrt(a))+1
    for i in range(3,rangeEnd,2):
        if a % i == 0:
            return False
    return True

print(isPrime(48112959837082048697))