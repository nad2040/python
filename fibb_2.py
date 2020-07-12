import time

def fibb2(n):
    if n >= 3:
        fib2 = fibb2(n-1)
        fib2.append(fib2[-1]+fib2[-2])
        return fib2
    if n == 2:
        return [1,1]
    if n == 1:
        return [1]
'''
begin = time.clock()
print fibb2(200)
end = time.clock()
print end - begin
'''
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

n = 100
fiblist = fibb3(n)
for i in range(n):
    print("%s\t%s" % (fiblist[i], 2**i))


