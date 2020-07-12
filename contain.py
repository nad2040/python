def contain(n,m):
    while (n > 0):
        print(n, n%10, n/10)
        if n % 10 == m:
            return True
        n = n / 10
    return False
        
print(contain(98888888, 5))


mul = 1
num = 999

while (contain(num * mul, 9)):
    mul = mul + 1

print(mul)