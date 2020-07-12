def factorial(n):
    fac = 1
    for i in range(1,n+1):
        fac = fac*i
    return fac

def fibb3(n) :
    if n == 1:
        return [1]
    elif n == 2:
        return [1,1]
    else:
        solution = [1,1]
        for i in range(2,n):
            solution.append(solution[-2]+solution[-1])
        return solution

def fac(n):
    if n == 1:
        return 1
    else:    
        return n*fac(n-1)

for i in range(1,30):
    print((fac(i), 2**i, fibb3(i)[-1]))

    