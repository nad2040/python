import random
from listtotuple import minmax

def inRange(num, low, high):
    return low < num <= high

def histgram(n, bucket):
    count = [0]*bucket
    bwidth = float(1/bucket)
    for num in (random.random() for i in range(n)):
        for t in range(bucket):
            if inRange(num, t*bwidth, (t+1)*bwidth) :
                count[t] += 1
    return tuple(count)

print(minmax(histgram(10000,11)))