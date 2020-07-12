from random import shuffle
from random import randint

x=[1,1,2,2,3,3,4,4,5,5,6,6]
count=0

for i in xrange(1000000):
	shuffle(x);
	if x[randint(0,5)] + x[randint(6,11)] == 7:
		count += 1
print count
