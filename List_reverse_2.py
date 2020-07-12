def reverse_list(l):
    if len(l) <= 1:
        return l
    else:
        mylist = reverse_list(l[1:])
        mylist.append(l[0])
        return mylist

print(reverse_list([1,2]))
print(reverse_list([1]))
print(reverse_list([1,2,3,4,1]))
l = list(range(20))
print(reverse_list(l))
