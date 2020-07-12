def fibb(n):
    fib=[]
    if n == 1:
        fib.append(1)
        return fib
    if n == 2:
        fib.append(1)
        fib.append(1)
        return fib
    for i in range(3, n+1):
        fib=fibb(i-1)
        fib.append(fib[-2]+fib[-1])
    return fib

def fib2(n):
    if n >= 3:
        tmp = fib2(n-1)
        tmp.append(tmp[-1]+tmp[-2])
        return tmp
    if n == 2:
        return [1,1]
    if n == 1:
        return [1]
        
print(fibb(11))
print(fib2(11))