count=0
for a in range(1,8):
    for b in range(1,8):
        for c in range(1,8):
            for d in range(1,8):
                if (3*a*b*c + 4*a*b*d + 5*b*c*d) % 2 == 0:
                    #print a, b, c, d
                    count = count + 1
print(count)