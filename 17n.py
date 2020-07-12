def multiple(n,a):
    mylist = []
    for i in range(0,a+1):
        if i % n == 0:
            mylist.append(i)
    return mylist

def notmultiple(n,a):
    mylist = [list(range(0,a+1))]
    for i in mylist:
        if i % n == 0:
            mylist.remove(i)
    return mylist

print(multiple(17,300))

print(multiple(3,100))
print(multiple(5,100))
print(multiple(3*5,100))
print(notmultiple(3,100))