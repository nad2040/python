def range_multiple(n,a):
    mylist = []
    for i in a:
        if i % n == 0:
            mylist.append(i)
    return mylist 

def multiple(n,a):
    mylist = []
    for i in range(0,a+1):
        if i % n == 0:
            mylist.append(i)
    return mylist

def notmultiple(m,n,a):
    mylist = []
    mylist = list(range(0,a+1))
    for i in range(0,a+1):
        if i % n == 0 or i % m == 0:
            mylist.remove(i)
    return mylist

print(multiple(17,300))

print(len(multiple(3,100)))
print(len(multiple(5,100)))
print(len(multiple(3*5,100)))
print(len(notmultiple(3,5,100)))
print(multiple(3,100))
print(multiple(5,100))
print(multiple(3*5,100))
print(notmultiple(3,5,100))

print(len(range_multiple(11,list(range(10000,100000)))))