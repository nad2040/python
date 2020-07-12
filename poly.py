import turtle

turtle.setup(800,800)                # Determine the window size
wn = turtle.Screen()                 # Get a reference to the window
Dan = turtle.Turtle()

def polygon(n,l):
    for i in range(n):
        Dan.forward(l)
        Dan.left(360.00/n)

def move(x, y):
    Dan.penup()
    Dan.goto(x,y)
    Dan.pendown()

def Quit():
    wn.bye()

wn.onkey(polygon(3, 150), "3")
wn.onkey(polygon(4, 150), "4")
wn.onkey(polygon(5, 150), "5")
wn.onkey(polygon(6, 150), "6")
wn.onkey(polygon(7, 150), "7")
wn.onkey(polygon(8, 150), "8")
wn.onkey(polygon(9, 150), "9")
wn.onkey(polygon(10, 150), "a")

wn.onkey(Quit, "q")

wn.onclick(move)

wn.listen()
turtle.mainloop()
