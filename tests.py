
'''
fruit = "banana"
count = 0
for ch in fruit:
    if ch == "a":
        count = count + 1

print(count) 

def myfind(str, ch) :
    for i in range(len(str)):
        if ch == str[i]:
            return i
    return -1

print(myfind(fruit,'a'))
print(myfind(fruit,'b'))
print(myfind(fruit,'c'))
'''
import string
print(string.ascii_lowercase)
print(string.ascii_uppercase)
print(string.ascii_letters)