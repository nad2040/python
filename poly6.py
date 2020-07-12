from math import cos,sin,pi
from turtle import Screen, Turtle

wn = Screen()
wn.setup(width=800, height=800, startx=0, starty=0)
Dan = Turtle()
Dan.ht()

xypairs = [(0,0), (-20, 50), (10,40)]

def rotateT(theta) :
    return [[cos(theta), sin(theta), 0], [-sin(theta), cos(theta), 0], [0, 0, 1]]

def moveT(deltaX, deltaY) :
    return [[1,0,0],[0,1,0],[deltaX,deltaY,1]]
    
def scaleT(k) :
    return [[k,0,0],[0,k,0],[0,0,1]]

def multMatrix(T1, T2):
    return [[
            T1[0][0]*T2[0][0]+T1[0][1]*T2[1][0]+T1[0][2]*T2[2][0],
            T1[0][0]*T2[0][1]+T1[0][1]*T2[1][1]+T1[0][2]*T2[2][1],
            T1[0][0]*T2[0][2]+T1[0][1]*T2[1][2]+T1[0][2]*T2[2][2],
            ],
            [
            T1[1][0]*T2[0][0]+T1[1][1]*T2[1][0]+T1[1][2]*T2[2][0],
            T1[1][0]*T2[0][1]+T1[1][1]*T2[1][1]+T1[1][2]*T2[2][1],
            T1[1][0]*T2[0][2]+T1[1][1]*T2[1][2]+T1[1][2]*T2[2][2],
            ],
            [
            T1[2][0]*T2[0][0]+T1[2][1]*T2[1][0]+T1[2][2]*T2[2][0],
            T1[2][0]*T2[0][1]+T1[2][1]*T2[1][1]+T1[2][2]*T2[2][1],
            T1[2][0]*T2[0][2]+T1[2][1]*T2[1][2]+T1[2][2]*T2[2][2],
            ]]
            
def transformT() :
    return multMatrix(multMatrix(rotateT(10*pi/180), moveT(10,10)), scaleT(1.00618))

def transformXY(pairxy, T) :
    newX = pairxy[0]*T[0][0] + pairxy[1]*T[1][0] + T[2][0]
    newY = pairxy[0]*T[0][1] + pairxy[1]*T[1][1] + T[2][1]
    return (newX, newY)
    
def newXYpairs(xypairs, T) :
    return [transformXY(xypairs[0], T), transformXY(xypairs[1], T), transformXY(xypairs[2], T)]


def drawXY(xypairs) :
    wn.tracer(0,0)
    Dan.pu()
    Dan.goto(xypairs[0])
    Dan.pd()
    Dan.goto(xypairs[1])
    Dan.goto(xypairs[2])
    Dan.goto(xypairs[0])
    wn.update()

def move(x, y):
    Dan.penup()
    Dan.goto(x,y)
    Dan.pendown()

def Quit():
    wn.bye()

def draw():
    shape = xypairs
    for _ in range(144):
        drawXY(shape)
        shape = newXYpairs(shape, transformT())
        Dan.clear()

wn.onkey(Quit, "q")
wn.onkey(draw, "d")
wn.onclick(move)

wn.listen()
wn.mainloop()