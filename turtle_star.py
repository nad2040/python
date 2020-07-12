import time
import turtle
wn = turtle.Screen()
Dan = turtle.Turtle()

def turtle_star(length):
    
    Dan.left(65)
    Dan.forward(150)
    Dan.right(144)
    Dan.forward(150)
    Dan.right(144)
    Dan.forward(150)
    Dan.right(144)
    Dan.forward(150)
    Dan.right(144)
    Dan.forward(150)
    
for length in range(50, 200):
    turtle_star(length)
    Dan.goto(length-40, length-40)
    
Dan.hideturtle()

time.sleep(60)