#print [x for x in range(100) if x % 3 == 0]
#print [(x,y) for x in range(100) for y in range(100) if x + y > x * y]

x = [[]]
y = [1,2,3]
x = [a+[b] for a in x for b in y]
x = [a+[b] for a in x for b in y]
x = [a+[b] for a in x for b in y]
x = [a+[b] for a in x for b in y]

print(x)
