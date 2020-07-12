import time

'''
for i in range(10102):
    if i % 21 == 0:
        print i
'''    

def zero_one(numstr):
    for c in numstr:
        if c != '0' and c != '1':
            return False
    return True

t0 = time.clock()    
for i in range(1,1000000):
    if zero_one(str(i * 42)):
        print i * 42  
t1 = time.clock()
print t1-t0

def onlyZeroOne(num):
    while num > 0:
        if num % 10 > 1:
            return False
        num = num/10
    return True

t0 = time.clock()    
for i in range(1,1000000):
    if onlyZeroOne(i * 42):
        print i * 42
t1 = time.clock()
print t1 - t0

t0 = time.clock()   
print [i*42 for i in range(1000000) if onlyZeroOne(i*42)]
t1 = time.clock()
print t1 - t0