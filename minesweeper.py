#!/usr/bin/env python3
import random, os

def clear():
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')

class Minesweeper:
        
    random.seed(os.urandom(4))

    def __init__(self):
        self.n_row = self.n_col = self.area = self.mine_density = self.mine_count = 0
        self.lost = self.won = False
        self.board = self.coords = self.display = self.visited = []

    def getdimension(self):
        print("Input dimensions: row col")
        self.n_row, self.n_col = [int(x) for x in input().split()]
        self.n_row = min(self.n_row, 50)
        self.n_col = min(self.n_col, 50)
        self.area = self.n_row * self.n_col

        if self.n_row == 0 or self.n_col == 0: print("input valid size"); return False
        else: return True

    def getdifficulty(self):
        difficulty = int(input("Choose a difficulty: \n0 - easy\n1 - medium\n2 - hard/difficult\n"))
        match difficulty:
            case 0:
                self.mine_density = random.uniform(0.05,0.1)
                self.mine_count = int(self.mine_density * self.area)
            case 1:
                self.mine_density = random.uniform(0.1,0.15)
                self.mine_count = int(self.mine_density * self.area)
            case 2:
                self.mine_density = random.uniform(0.15,0.25)
                self.mine_count = int(self.mine_density * self.area)
            case _:
                print("input valid difficulty"); return False
                
        print("Size: {} rows by {} cols\n{} mines with a density of {}%".format(self.n_row, self.n_col, self.mine_count, int(self.mine_density * 100)))
        return True

    def initboards(self):
        self.board = [['0']*self.n_col for _ in range(self.n_row)]
        self.coords = [(i, j) for i in range(self.n_row) for j in range(self.n_col)]
        self.display = [['#']*self.n_col for _ in range(self.n_row)]
        self.visited = [[False]*self.n_col for _ in range(self.n_row)]

    def createmines(self):
        neighbors = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
        for _ in range(self.mine_count):
            x,y = coord = random.choice(self.coords)
            
            # check if this location has 5 or more mines around it
            count = 0
            for (i,j) in neighbors:
                if x+i in range(self.n_row) and y+j in range(self.n_col) and self.board[x+i][y+j] == '*':
                    count += 1
            if count < 5:
                self.board[x][y] = '*'
            else:
                continue
            # increment value around mines
            for (i,j) in neighbors:
                if x+i in range(self.n_row) and y+j in range(self.n_col):
                    self.board[x+i][y+j] = chr(ord(self.board[x+i][y+j]) + 1) if self.board[x+i][y+j] != '*' else self.board[x+i][y+j]
            self.coords.remove(coord)

    def flag(self, x, y, flag=True):
        if (self.display[x][y] in "f#"):
            if self.display[x][y] == '#' and flag:
                self.display[x][y] = 'f'
                return True
            elif self.display[x][y] == 'f' and not flag:
                self.display[x][y] = '#'
                return True
            else:
                return False
        else:
            return False

    def clrarea(self, x, y):
        self.visited[x][y] = True
        self.display[x][y] = ' ' if self.board[x][y] == '0' else self.board[x][y]
        neighbors = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
        for r,c in neighbors:
            if (self.board[x][y] == '0' and x+r in range(self.n_row) and y+c in range(self.n_col) and not self.visited[x+r][y+c]):
                self.clrarea(x+r, y+c)

    def click(self, x, y):
        if self.display[x][y] == '#':
            if self.board[x][y] == '*':
                self.display[x][y] = self.board[x][y]
                self.lost = True
                return True
            elif self.board[x][y] == '0':
                self.visited = [[False]*self.n_col for _ in range(self.n_row)]
                self.clrarea(x, y)
                return True
            else:
                self.display[x][y] = self.board[x][y]
                return True
        else:
            return False
        
    def play(self):
        self.printboard(self.display)

        print("Type your command: <flag|unflag|click> r c")
        move,r,c = [_ for _ in input().split()]
        r=int(r)
        c=int(c)
        if r not in range(self.n_row) or c not in range(self.n_col):
            print("out of bounds")
            return False

        match move:
            case "flag":
                if self.flag(r,c):
                    print("flagged row {} col {}".format(r,c))
                    return True
                else:
                    print("couldn't flag row {} col {}".format(r,c))
                    return False
            case "unflag":
                if self.flag(r,c,False):
                    print("unflagged row {} col {}".format(r,c))
                    return True
                else:
                    print("couldn't unflag row {} col {}".format(r,c))
                    return False
            case "click":
                if self.click(r,c):
                    print("clicked row {} col {}".format(r,c))
                    return True
                else:
                    print("couldn't click row {} col {}".format(r,c))
                    return False
            case _:
                print("invalid move")
                return False

    def checkwin(self):
        count = len([c for c in sum(self.display, []) if c in "#f"])
        return count == self.mine_count

    def win(self):
        self.printboard(self.display)
        print("You won")

    def lose(self):
        self.printboard(self.display)
        print("You hit a mine")

    def printboard(self, board):
        print(("\n-" + "+-"*(len(board[0])-1) + '\n').join(['|'.join(["{}".format(cell) for cell in row]) for row in board]))


m = Minesweeper()
while True:
    if not m.getdimension(): continue
    if not m.getdifficulty(): continue
    m.initboards()
    m.createmines()
    
    while True:
        m.play()
        if m.lost:
            break
        if m.checkwin():
            m.won = True
            break

    if m.won:
        m.win()
    else:
        m.lose()

    c = input("Do you wish to keep playing (Y/n): ")
    if len(c)==0 or c in "yY": continue
    else: break