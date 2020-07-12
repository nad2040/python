def mysort(mylist):
    for j in range(len(mylist)):
        for i in range(j, len(mylist)):
            if mylist[j] > mylist[i]:
                mylist[j], mylist[i] = mylist[i], mylist[j]

t = [77, 42, 85, 12, 25, 5, 66]

mysort(t)

print(t)



'''
def printlist(l):
    length = len(l)
    for i in range(length):
        #print l.pop(0)
        print l[i]

printlist(t)

print len(t)

#sort(t)

'''