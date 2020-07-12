def inrange(x,y,n):
    if 0 <= x < n and 0 <= y < n:
        return True

def validmove(board, row, col, n):
    for i in range(1, row+1):
        if inrange(row-i, col, n) and board[row-i][col] == 1:
            return False
        if inrange(row-i, col+i, n) and board[row-i][col+i] == 1:
            return False
        if inrange(row-i, col-i, n) and board[row-i][col-i] == 1:
            return False
    return True

def placeQueen(row, board, n):
    if row == n:
        print(board)
        return
    for col in range(n):
        board[row][col] = 1
        if validmove(board, row, col, n):
            placeQueen(row+1, board, n)
        board[row][col] = 0

n = 8
row = [0]*n
board = [row[:] for i in range(n)]

placeQueen(0, board, n)