
def server(f):
    tasks = []
    value = None
    while True:
        batch = yield value
        value = None
        if batch is not None:
            tasks.extend(batch)
        else:
            if tasks:
                args = tasks.pop(0)
                value = f(*args)


s = server(str)
s.send(None)
s.send([(1,),(2,),(3,)])
print(next(s))
print(next(s))
print(next(s))
