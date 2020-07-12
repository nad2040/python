import random

a = 1000
n = []

for i in range(a):
    n.append(random.random())

w = []
x = []
y = []
z = []
 
for i in range(len(n)):
    if 0.0<n[i]<0.25:
        w.append(n[i])
    if 0.25<n[i]<0.5:
        x.append(n[i])
    if 0.5<n[i]<0.75:
        y.append(n[i])
    if 0.75<n[i]<1.0:
        z.append(n[i])

print(len(w),len(x),len(y),len(z))