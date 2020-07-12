def increasing(num):
    if num < 10:
        return True
    elif num < 100:
        return num % 10 > num / 10
    else:
        last = num % 10
        num = num / 10
        sec = num % 10
        if last > sec:
            return increasing(num)
        else:
            return False
        
def decreasing(num):
    if num < 10:
        return True
    elif num < 100:
        return num % 10 < num / 10
    else:
        last = num % 10
        num = num / 10
        sec = num % 10
        if last < sec:
            return decreasing(num)
        else:
            return False

#print increasing(11234)
#print decreasing(54321)

count = 0
for i in range(123456789):
    if increasing(i):
        count = count + 1
        
print(count)

count = 0
for i in range(9876543210):
    if decreasing(i):
        count = count + 1
        
print(count)
'''
def monotonous(number):
    digits = [int(d) for d in str(number)]
    if len(digits) != len(set(digits)):
        return False
    sorted_digits = sorted(digits, reverse=True)
    num_sorted1 = int(''.join(map(str,sorted_digits)))
    reverse_sorted_digits = sorted(digits)
    num_sorted2 = int(''.join(map(str,reverse_sorted_digits)))
    return num_sorted1 == number or num_sorted2 == number
'''    
    
'''       
print monotonous(5)
print monotonous(57)
print monotonous(548)
print monotonous(5547)
print monotonous(5747)
print monotonous(12789)
'''
'''
count = 0;
for i in range(1000):
    if monotonous(i):
        count = count + 1
        
print count
'''

#results = [i for i in range(100) if monotonous(i)]

#print results
#print len(results)
