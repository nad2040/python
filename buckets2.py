import random

count0 =  count1 = count2 = count3 = 0
for i in range(10000):
    num = random.random()
    if 0 <= num <= 0.25:
        count0 += 1
    elif 0.25 < num <= 0.5:
        count1 += 1
    elif 0.5 < num <= 0.75:
        count2 += 1
    else:
        count3 += 1

print(count0,count1,count2,count3)

