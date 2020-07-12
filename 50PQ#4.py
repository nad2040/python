from random import randint
"""
50 prob problems #4
"""
def till6() :
    count = 1
    while (randint(1,6) !=6) :
        count += 1
    return count
    
def avg6(N) :
    sum = 0
    for i in range(N) :
        sum += till6()
    return 1.0 * sum / N
    
print(avg6(1000))

print(avg6(100000))