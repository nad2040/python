import turtle
Dan = turtle.Turtle()

def turtle_square(length):
    for i in range(4):
        Dan.forward(length)
        Dan.left(90)

Dan.penup()
Dan.goto(-320, 240)
Dan.pendown()

for i in range(8):
    for i in range(8):
        turtle_square(80)
        Dan.forward(80)
    Dan.right(180)
    Dan.forward(640)
    Dan.left(90)
    Dan.penup()
    Dan.forward(80)
    Dan.pendown()
    Dan.left(90)

turtle.mainloop()
