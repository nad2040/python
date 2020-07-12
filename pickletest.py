import pickle

# mystring = "{} {} {}".format(([1,1,1]),(1,2,2),({1:1,2:2}))
# print(mystring)

l = [5, 4.0, 'hello']
t = ("bye", 29, 5.2)
d = {1:1.0,2:2.0,"hi":"bye"}

def hello(world):
    return 'hello' + world

with open("tmp/test4.dat","wb") as f:
    pickle.dump(l, f)
    pickle.dump(t, f)
    pickle.dump(d, f)
    pickle.dump(hello, f)
    with open("pickletest.py", "r") as src:
        pickle.dump(src.read(), f)

with open("tmp/test4.dat","rb") as f:
    while True:
        try:
            x = pickle.load(f)
            print(type(x))
            if callable(x):
                print(x('world'))
                print(x.__code__)
            elif isinstance(x,str):
                #eval(x) 
                print(x)
            else:
                print(x)
        except EOFError:
            print("No more elements")
            break