def reverse(l) :
    if len(l) <= 1:
        return l
    tmp = reverse(l[1:])
    tmp.append(l[0])
    return tmp
        
print(reverse([]))
print(reverse([5]))
print(reverse([100]))
alist = [3,5,6,20,17,42,111,23]
blist = reverse(alist)
alist.reverse()
print(blist)
assert(set(alist) == set(blist))


def list_reverse(l):
    reversed_list = []
    for i in l:
        prepend(reversed_list,i)
    return reversed_list

def prepend(l,x):
    l.insert(0,x)
    return l
    
ll = prepend([3,5,7],13)
for i in range(len(ll)-1,-1,-1):
    print(ll[i])

print(list_reverse([3,5,9,13]))
print(list_reverse([]))
print(list_reverse([13]))