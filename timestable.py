def multiple(n):
    str = ''
    for i in range(1,n+1):
        str = str + ("{} * {} = {}\t").format(i, n, i*n)
    return str

def timestable(n):
    for i in range(1,n+1):
        print(multiple(i))

timestable(9)
#print(multiple(3))
