mystring = """        North of 53. A magic phrase. Spoken, mumbled or thought
    inwardly by thousands of souls venturing northward. An
    imaginary line, shown only on maps and labelled 53 degrees.
    It's presence indicated to highway travellers by road side
    signs.
        A division of territory as distinct in the mind as any
    international border.
        If you have not been "North of 53", you have not been
    north!"""

f = open("tmp/test3.dat","w")
f.write(mystring)

f = open("tmp/test3.dat","r")
numlines = len(f.readlines())

f = open("tmp/test3.dat","r")
numchar = len(f.read()) - numlines + 1

print(numlines)
print(numchar)

with open("tmp/test3.dat", "r") as f:
    count = 0
    tc = 0
    for line in f:
        count += 1
        tc += len(line)
print(count)
print(tc)


