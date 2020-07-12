def rect(row, col):
    for i in range(row+1):
        print(' '*20+chr(i+32)*col)
        
def pgramn(row, col):
    for i in range(row+1):
        print(' '*(row-i)+chr(i+32)*col)
        
pgramn(1, 1)
pgramn(1, 2)
pgramn(2, 3)
pgramn(3, 5)
pgramn(5, 8)

def pgramp(row, col):
    for i in range(row+1):
        print(' '*i+chr(i+32)*col)

pgramp(1, 1)
pgramp(1, 2)
pgramp(2, 3)
pgramp(3, 5)
pgramp(5, 8)
pgramn(5, 8)