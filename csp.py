answers = []
for i in range(0,10000):
    if len(set(str(i))) == len(str(i)) and int("{:04d}".format(i)) // 1000 != 0 and int("{:04d}".format(i)) %2 == 0:
        answers.append(i)
print(len(answers))