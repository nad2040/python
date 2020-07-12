class ChessBoard:
    def __init__(self):
        row = [0]*8
        self.board = [row[:] for i in range(8)]
    def __str__(self):
        return str(self.board)
    def placeQueen(self, i, j):
        self.board[i] = [0]*8
        self.board[i][j] = 1
    def goodLocation(self, x, y):
        return 0 <= x < 8 and 0 <= y < 8
    def validDiag(self, x, y):
        for i in range(1,8):
            xnew = x + i
            ynew = y + i
            if self.goodLocation(xnew, ynew) and self.board[xnew][ynew] == 1:
                return False
            xnew = x - i
            ynew = y - i
            if self.goodLocation(xnew, ynew) and self.board[xnew][ynew] == 1:
                return False
            xnew = x + i
            ynew = y - i
            if self.goodLocation(xnew, ynew) and self.board[xnew][ynew] == 1:
                return False
            xnew = x - i
            ynew = y + i
            if self.goodLocation(xnew, ynew) and self.board[xnew][ynew] == 1:
                return False
        return True
 
    def validDiagonal(self, i, j):
        sum = 0
        for n in range(-8,8):
            x = i+n
            y = j+n
            if 0 <= x < 8 and 0 <= y < 8:
                sum += self.board[x][y]
            if sum != 1:
                return False
        sum = 0
        for n in range(-8,8):
            x = i+n
            y = j-n
            if 0 <= x < 8 and 0 <= y < 8:
                sum += self.board[x][y]
            if sum != 1:
                return False
    def isValid(self):
        #check no duplicate in col
        for col in range(8):
            sum = 0
            for row in range(8):
                sum += self.board[row][col]
            if sum != 1:
                return False
        return True
    def isValidDiag(self):
        #check no duplicate in diagnol
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == 1:
                    if not self.validDiag(row, col):
                        return False       
        return True

#b = ChessBoard()
#b.placeQueen(2,2)
#print(b)

# import random

# def placeQueenAtRow(b, row):
#     b.placeQueen(row, random.randint(0,7))

# for count in range(10000000):
#     b = ChessBoard()
#     for row in range(8):
#         placeQueenAtRow(b, row)
#     if b.isValid() and b.isValidDiag():
#         print(b)

count = 0
for c0 in range(8):
    board = ChessBoard()
    board.placeQueen(0,c0)
    for c1 in range(8):
        board.placeQueen(1,c1)
        for c2 in range(8):
            board.placeQueen(2,c2)
            for c3 in range(8):
                board.placeQueen(3,c3)
                for c4 in range(8):
                    board.placeQueen(4,c4)
                    for c5 in range(8):
                        board.placeQueen(5,c5)
                        for c6 in range(8):
                            board.placeQueen(6,c6)
                            for c7 in range(8):
                                board.placeQueen(7,c7)
                                count += 1
                                # if count % 1000 == 0:
                                #     print(count)
                                if board.isValid() and board.isValidDiag():
                                    print(board)
                                