#!/usr/bin/env python3

class Sudoku:
    def __init__(self,board=None):
        if board is not None:
            self.board = board
        else:
            self.board = [[' ']*9 for _ in range(9)]

    def printboard(self):
        print(("\n-" + "+-"*(len(self.board[0])-1) + '\n').join(['|'.join(["{}".format(cell) for cell in row]) for row in self.board]))

    def boxnum(self,r,c):
        return 3*((r-1)//3) + (c-1)//3 + 1

    def setgivens(self,givens):
        '''
        givens is a list of tuples in the format (row,col,number) that describe where each given digit is.
        row and col are from 1 to 9 like in traditional sudoku notation and the function automatically deals with index number
        '''
        for r,c,num in givens:
            self.board[r-1][c-1] = num

    def checkrow(self, row, num):
        items = [self.board[r][c] for r in range(9) if r == row-1 for c in range(9)]
        return str(num) not in items

    def checkcol(self, col, num):
        items = [self.board[r][c] for c in range(9) if c == col-1 for r in range(9)]
        return str(num) not in items

    def checkbox(self, box, num):
        items = [self.board[r][c] for r in range(9) for c in range(9) if 3*(r//3) + c//3 == box-1]
        return str(num) not in items

    def canplace(self, r, c, num):
        return self.checkrow(r, num) and self.checkcol(c, num) and self.checkbox(self.boxnum(r,c), num)

    def place(self, r, c, num):
        # print("Placing {} at r{} c{}".format(num,r,c))
        self.board[r-1][c-1] = str(num)


    def solve(self):
        for r in range(1,10):
            for c in range(1,10):
                if self.board[r-1][c-1] == ' ':
                    for num in range(1,10):
                        # print("Try to place {} at r{} c{}".format(num,r,c))
                        if self.canplace(r,c,num):
                            self.place(r,c,num)
                            if not self.solve(): self.board[r-1][c-1] = ' '
                    return False
        self.printboard()
        print()
        return True



s = Sudoku([
    [' ',' ',' ','2','6',' ','7',' ','1'],
    ['6','8',' ',' ','7',' ',' ','9',' '],
    ['1','9',' ',' ',' ','4','5',' ',' '],
    ['8','2',' ','1',' ',' ',' ','4',' '],
    [' ',' ','4','6',' ','2','9',' ',' '],
    [' ','5',' ',' ',' ','3',' ','2','8'],
    [' ',' ','9','3',' ',' ',' ','7','4'],
    [' ','4',' ',' ','5',' ',' ','3','6'],
    ['7',' ','3',' ','1','8',' ',' ',' ']
])

# s.setgivens([
    # (1,4,'2'),
    # (1,5,'6'),
    # (1,7,'7'),
    # (1,9,'1'),
    # (2,1,'6'),
    # (2,2,'8'),
    # (2,5,'7'),
    # (2,8,'9'),
    # (3,1,'1'),
    # (3,2,'9'),
    # (3,6,'4'),
    # (3,7,'5'),
    # (4,1,'8'),
    # (4,2,'2'),
    # (4,4,'1'),
    # (4,8,'4'),
    # (5,3,'4'),
    # (5,4,'6'),
    # (5,6,'2'),
    # (5,7,'9'),
    # (6,2,'5'),
    # (6,6,'3'),
    # (6,8,'2'),
    # (6,9,'8'),
    # (7,3,'9'),
    # (7,4,'3'),
    # (7,8,'7'),
    # (7,9,'4'),
    # (8,2,'4'),
    # (8,5,'5'),
    # (8,8,'3'),
    # (8,9,'6'),
    # (9,1,'7'),
    # (9,3,'3'),
    # (9,5,'1'),
    # (9,6,'8')
# ])

s.printboard()
print()
s.solve()
