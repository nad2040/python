from math import *
import turtle

turtle.setup(800,800)                # Determine the window size
wn = turtle.Screen()                 # Get a reference to the window
Dan = turtle.Turtle()

xypairs = [(0,0), (0, 100), (100,0)]
theta = 30 * pi / 180
T = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]

def rotateXY(pairxy) :
    newX = pairxy[0]*T[0][0] + pairxy[1]*T[0][1]
    newY = pairxy[0]*T[1][0] + pairxy[1]*T[1][1]
    return (newX, newY)

newXYpairs = [rotateXY(xypairs[0]), rotateXY(xypairs[1]), rotateXY(xypairs[2])]

def drawXY() :
    Dan.pu()
    Dan.goto(xypairs[0])
    Dan.pd()
    Dan.goto(xypairs[1])
    Dan.goto(xypairs[2])
    Dan.goto(xypairs[0])
    
def drawXYNew() :
    Dan.pu()
    Dan.goto(newXYpairs[0])
    Dan.pd()
    Dan.goto(newXYpairs[1])
    Dan.goto(newXYpairs[2])
    Dan.goto(newXYpairs[0])
    
def move(x, y):
    Dan.penup()
    Dan.goto(x,y)
    Dan.pendown()

def Quit():
    wn.bye()

wn.onkey(Quit, "q")
wn.onkey(drawXY, "d")
wn.onkey(drawXYNew, "r")
wn.onclick(move)

wn.listen()
turtle.mainloop()
