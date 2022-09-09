def fibonacci(n):
    if n == 0:
        return 1
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(50))

memory = {0:1, 1:1}
def fastfib(n):
    if n in memory:
        return memory[n]
    else:
        memory[n] = fastfib(n-1) + fastfib(n-2)
        return memory[n]

#print(fastfib(50))
