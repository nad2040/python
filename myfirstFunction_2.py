def fac(n):
    product = 1
    for i in range(1, n+1):
        print("%s * %s" % (product, i))
        product = product * i
    return product

def fac2(n):
    if n == 1: 
        print("n is 1, return 1")
        return 1
    print("%s * factorial(%s)" % (n, n-1))
    result = n * fac2(n-1)
    return result
    
def fib(n):
    if n == 1:
        result = 1
    if n == 2:
        result = 1
    if n >= 3:
        result = fib(n-1) + fib(n-2)
    return result
    
    
print(fac(10))
print(fac2(10))
print(fib())