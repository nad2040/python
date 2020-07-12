class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'

p = Point(5, 6)
p.x = 7
p.y = 90
print(type(p))
print(p)

def distance(pointA, pointB):
    return pointB.y - pointA.y + pointB.x - pointA.x

dis = distance(Point(3,4), Point(5,6))
print(dis)