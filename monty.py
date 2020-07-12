from random import *
setups = [(0,0,1), (0,1,0), (1,0,0)]

count = 0
for i in range(10000000):
    indexes = list(range(len(setups)))
    setup = choice(setups)
    first = choice(indexes)
    
    doors = [i for i in indexes if setup[i] == 0]
    if first in doors:
        doors.remove(first)
    opendoor = choice(doors)
    
    indexes.remove(first)
    indexes.remove(opendoor)
    switchto = indexes[0]
    '''
    print "setup: " + str(setup)
    print "first choice: " + str(first)
    print "doors we can open: " + str(doors)
    print "open door: " + str(opendoor)
    print "switch to: " + str(switchto)

    if setup[first]==1:
        print "first choice is a winner"
    elif setup[switchto]==1:
        print "switch is a winner"
    else:
        print "something is wrong"
    '''
    if setup[switchto] == 1:
        count = count + 1
print(count)