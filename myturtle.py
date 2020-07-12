import time
import turtle
wn = turtle.Screen()
Dan = turtle.Turtle()

def turtle_square(length):
    Dan.forward(length/2)
    Dan.left(90)
    Dan.forward(length)
    Dan.left(90)
    Dan.forward(length)
    Dan.left(90)
    Dan.forward(length)
    Dan.left(90)
    Dan.forward(length/2)
    
for length in range(50,200,10):
    turtle_square(length)
    Dan.left(10)
    
Dan.hideturtle()

#wn.mainloop()s
time.sleep(60)