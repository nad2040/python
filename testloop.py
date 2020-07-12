def printaline():
    answer = ''
    for i in range(11):
        answer = answer + str(i)
    print(answer)
    
for i in range(10):
    printaline()
    
def term(n):
    answer = ''
    for i in range(1, n+1):
        answer = answer + "%s * %s = %s\t" % (i, n, i * n)
    print(answer)

for i in range(1, 10):
    term(i)