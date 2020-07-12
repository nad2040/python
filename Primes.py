import time

def primes(n):
    bag = []
    mlist = list(range(2,n+1))
    while len(mlist) > 0:
        prime = mlist[0]
        bag.append(prime)
        for i in mlist:
            if i % prime == 0:
                mlist.remove(i)
    return bag
    
def myprime(n):
    col = list(range(2,n+1))
    idx = 0
    while idx < len(col):
        col = [x for x in col if x % col[idx] != 0 or x == col[idx]]
        idx = idx + 1
    return col

begin = time.clock()    
myprime(2**15)
end = time.clock()
print(end-begin)

begin = time.clock()
primes(2**15)
end = time.clock()
print(end-begin)
    
    
    
