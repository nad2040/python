def all_multiple(l, n):
    for i in l:
        if i%n != 0:
            return False
    return True

print(all_multiple([3,6,9,12,15,21], 3))

def all_divide(l, n):
    mylist = []
    for i in l:
        mylist.append(i/n)
    return mylist
    
print(all_divide([3,6,9,12,15,21], 3))

primes = [2,3,5,7,11,13,17,19]

def gcd(l):
    result = 1
    for i in range(10):
        for j in primes:
            if all_multiple(l,j):
                l = all_divide(l,j)
                result = result*j
    return result
    
print(gcd([22,33,77,88,99,121]))