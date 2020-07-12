def chessboard():
    row = [0]*8
    board = [row[:] for i in range(8)]
    return board

def goodLocation(x, y):
    return 0 <= x < 8 and 0 <= y < 8

def canPlaceQueen(board, row, col):
    #print(row, col)
    #print(board)
    for i in range(1,row+1):
        if goodLocation(row-i, col) and board[row-i][col] == 1:
            return False
        if goodLocation(row-i, col+i) and board[row-i][col+i] == 1:
            return False
        if goodLocation(row-i, col-i) and board[row-i][col-i] == 1:
            return False
    return True

# def placeQueen(row,board):
#     #print(row)
#     #print(board)
#     if row == 7:
#         for col in range(8):
#             if canPlaceQueen(board, row, col):
#                 board[row][col] = 1
#                 print(board)
#             else:
#                 board[row][col] = 0
#         return
#     for col in range(8):
#         if canPlaceQueen(board, row, col):
#             board[row][col] = 1
#             if not placeQueen(row+1,board):
#                 board[row][col] = 0

# def placeQueen(row,board):
#     for col in range(8):
#         if canPlaceQueen(board, row, col):
#             board[row][col] = 1
#             if row == 7:
#                 print(board)
#                 board[row][col] = 0
#                 return False
#             elif placeQueen(row+1,board):
#                 return True
#             else:
#                 board[row][col] = 0

def placeQueen(row, board):
    if row == 8:
        print(board)
    for col in range(8):
        if canPlaceQueen(board, row, col):
            board[row][col] = 1
            if placeQueen(row+1, board):
                pass
            else:
                board[row][col] = 0
    return False



    
    
board = chessboard()        
placeQueen(0,board)
#print(board)

'''
[1, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 1, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 1], 
[0, 0, 0, 0, 0, 1, 0, 0], 
[0, 0, 1, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 1, 0], 
[0, 1, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 1, 0, 0, 0, 0]
'''