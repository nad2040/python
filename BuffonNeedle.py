import math
import random

def needle(x,y,theta):
    newX = x+math.cos(theta)
    newY = y-math.sin(theta)
    return [x, y, newX, newY]

def success(needle):
    y1 = needle[1]
    y2 = needle[3]
    if y1 < 0 or y2 < 0:
        return 1
    if y1 > 10 or y2 > 10:
        return 1
    num1 = int(y1)
    num2 = int(y2)
    if num2 > num1 and num2 %2 == 0:
        return 1
    if num1 > num2 and num1 %2 == 0:
        return 1
    return 0

count = 1000000
sum = 0
for i in range(count):
    x = 10*random.random()
    y = 10*random.random()
    theta = math.pi*2*random.random()
    sample = needle(x,y,theta)
    sum = sum + success(sample)

print(count/sum)