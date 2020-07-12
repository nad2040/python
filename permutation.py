import itertools

numberList = [1,2,3,4,5,6,7,8,9,0]
#question = 'SEND + MORE = MONEY'
#question = 'YOYO - POP = POP'
question = 'abc - de = fh'

def genList(mylist, number):
    return list(itertools.permutations(mylist, number))

def uniqueChars(expression):
    return list(set(expression.replace(' ','').replace('+', '').replace('=', '')))

def toNumber(str, dict):
    sum = 0
    for i in range(len(str)):
        if str[i] in dict:
            sum = sum * 10 + dict[str[i]]
    return sum

#print(toNumber("abc", {'a':3, 'b':4, 'c':5}))

def validExpression(expression):
    l = expression.split()
    if len(l) != 5:
        print('not valid eq')
        return False
    elif l[1] not in ['+', '-']:
        print("invalid operator")
        return False
    elif l[3] != '=':
        print("invalid operator")
        return False
    else:
        print('first operand is ', l[0])
        print('operator is ', l[1])
        print('second operand is ', l[2])
        print('result is ', l[4])
    return True

def validTranslation(string, num):
    if len(string) != len(str(num)):
        return False
    return True

#print(validTranslation('abc', 345))

def evaluateExpression(items, dict):
    firstOpt = toNumber(items[0], dict)
    if not validTranslation(items[0], firstOpt):
        return
    secondOpt = toNumber(items[2], dict)
    if not validTranslation(items[2], secondOpt):
        return
    result = toNumber(items[4], dict)
    if not validTranslation(items[4], result):
        return

    op = items[1]
    if op == '+' and firstOpt + secondOpt == result:
        print(firstOpt, op, secondOpt, '=', result)
    elif op == '-' and firstOpt - secondOpt == result:
        print(firstOpt, op, secondOpt, '=', result)
        
def cryptarithm(expression):

    if not validExpression(question):
        print("not valid expression")
        exit(0)

    charList = uniqueChars(question)
    #print(charList)
    if len(charList) > 10:
        print("too many unique letters:", len(charList))
        exit(0)

    elems = question.split()

    for item in genList(numberList, len(charList)):
        evaluateExpression(elems, dict(zip(charList, item)))
 
    print('done')

cryptarithm(question)


'''
def assignvalue(l):
    for permutation in l:
        for val in range(len(permutation)):
            digits = list(itertools.combinations('1234567890',len(permutation)))
            #permutation[val] = digits[val]
            print(toNumber(digits[val]))
    return l

print(len(assignvalue(perms)))
'''