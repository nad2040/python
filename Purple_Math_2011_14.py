def to_digits(n):
    while (n > 0):
        print(n % 10)
        n = n / 10

def sum_digits(n):
    sum=0
    while (n > 0):
        sum = sum + n % 10
        n = n / 10
    return sum

for i in range(99930,100000):
    if i % sum_digits(i) == 0:
        print(i)