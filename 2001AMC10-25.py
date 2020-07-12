numlist=[]
for i in range(1,2002):
    if (i %3 == 0 or i %4 == 0) and i %5 != 0:
        numlist.append(i)

#print numlist
#print len(numlist)

print(len([x for x in range(1,2002) if (x%3==0 or x%4==0) and x%5!=0]))