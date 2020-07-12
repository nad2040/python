mylist = list(range(1,201))
mylistEven = list(range(2,201,2))
mylistOdd = list(range(1,201,2))

print(mylistEven)
print(mylistOdd)

print([x for x in range(1,201)])
print([2*x+1 for x in range(0,200)])
print([2*x for x in range(200)])

print([x for x in range(400) if x % 2 == 1])

    