'''
def hanoi(n,a,b,c):
    if n == 3:
        print move(1,1,3)
        print move(2,1,2)
        print move(1,3,2)
        print move(3,1,3)
        print move(1,2,1)
        print move(2,2,3)
        print move(1,1,3)
        
def move(n,towera,towerb):
    return [n,towera,towerb]

hanoi(3,1,2,3)
'''

def hanoii(n, source, helper, dest):
    if n == 1:
        print("move %d from %s to %s" % (n, source, dest))
    else:
        hanoii(n-1, source, dest, helper)
        print("move %d from %s to %s" % (n, source, dest))
        hanoii(n-1, helper, source, dest)

#hanoii(1, 'A', 'B', 'C')
#hanoii(2, 'A', 'B', 'C')
#hanoii(3, 'A', 'B', 'C')
#hanoii(4, 'A', 'B', 'C')
hanoii(5, 'A', 'B', 'C')