def sum_digits(n):
    sum=0
    while (n > 0):
        sum = sum + n % 10
        n = n / 10
    return sum
    
a = 1
for i in range(156):
    print(sum_digits(a), end=' ')
    if (i+1) % 12 == 0:
        print()
    a = a + sum_digits(a)
