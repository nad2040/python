def factorial(n) :
    p = 1;
    while n > 1 :
        newp = p * n
        sentence = 'The product of {} and {} is {}.'.format(p, n, newp)
        print(sentence)
        n = n - 1
        p = newp
    print(p)
    
factorial(6)

        