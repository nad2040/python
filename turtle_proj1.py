import turtle

turtle.setup(800,800)                # Determine the window size
wn = turtle.Screen()                 # Get a reference to the window
Dan = turtle.Turtle()

def equi_tri():
    Dan.forward(75)
    Dan.left(120)
    Dan.forward(150)
    Dan.left(120)
    Dan.forward(150)
    Dan.left(120)
    Dan.forward(75)

def square():
    Dan.forward(75)
    Dan.left(90)
    Dan.forward(150)
    Dan.left(90)
    Dan.forward(150)
    Dan.left(90)
    Dan.forward(150)
    Dan.left(90)
    Dan.forward(75)

def pent():
    Dan.forward(75)
    Dan.left(72)
    Dan.forward(150)
    Dan.left(72)
    Dan.forward(150)
    Dan.left(72)
    Dan.forward(150)
    Dan.left(72)
    Dan.forward(150)
    Dan.left(72)
    Dan.forward(75)

def move(x, y):
    Dan.penup()
    Dan.goto(x,y)
    Dan.pendown()

def Quit():
    wn.bye()

wn.onkey(equi_tri, "3")
wn.onkey(square, "4")
wn.onkey(pent, "5")
wn.onkey(Quit, "q")

wn.onclick(move)

wn.listen()
turtle.mainloop()
