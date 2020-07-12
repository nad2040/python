def stairs(steps,results):
    if steps == 1:
        return [x+[1] for x in results]
    elif steps == 2:
        return [x+[2] for x in results] + [x+[1,1] for x in results]
    else:
        return stairs(steps-1, [x+[1] for x in results]) + stairs(steps-2, [x+[2] for x in results])

print(stairs(5,[[]]))

x = [[]]
y = [1,2,3]
n = 6
for i in range(n):
    x = [a+[b] for a in x for b in y if sum(a+[b]) <= n]
    l = [c for c in x if sum(c) == n]
    if len(l) > 0:
        print(l)



