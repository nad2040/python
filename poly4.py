from math import *
import random
import turtle

turtle.setup(800,800)                # Determine the window size
wn = turtle.Screen()                 # Get a reference to the window
Dan = turtle.Turtle()

xypairs = [(0,0), (0, 100), (100,0)]

def rotateT() :
    theta = random.randint(1,31) * pi / 180
    return [[cos(theta), sin(theta), 0], [-sin(theta), cos(theta), 0], [0, 0, 1]]

def moveT() :
    deltaX = random.randint(1,21)
    deltaY = random.randint(1,21)
    return [[1,0,0],[0,1,0],[deltaX,deltaY,1]]

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
            
transformT = multMatrix(moveT(), rotateT())
transformT = multMatrix(transformT, moveT())

def transformXY(pairxy, T) :
    newX = pairxy[0]*T[0][0] + pairxy[1]*T[1][0] + T[2][0]
    newY = pairxy[0]*T[0][1] + pairxy[1]*T[1][1] + T[2][1]
    return (newX, newY)
    
def newXYpairs(xypairs, T) :
    return [transformXY(xypairs[0], T), transformXY(xypairs[1], T), transformXY(xypairs[2], T)]

def drawXY(xypairs) :
    Dan.pu()
    Dan.goto(xypairs[0])
    Dan.pd()
    Dan.goto(xypairs[1])
    Dan.goto(xypairs[2])
    Dan.goto(xypairs[0])
    
def move(x, y):
    Dan.penup()
    Dan.goto(x,y)
    Dan.pendown()

def Quit():
    wn.bye()

def draw():
    shape = xypairs
    for i in range(30):
        drawXY(shape)
        shape = newXYpairs(shape, transformT)

wn.onkey(Quit, "q")
wn.onkey(draw, "d")
wn.onclick(move)

wn.listen()
turtle.mainloop()
