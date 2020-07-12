import re

def tokenize(line):
    return [_f for _f in re.split(r'(\+|-|\*|/|\(|\))', "".join(line.split())) if _f]

def peek(tokens):
    print("peek:" + " ".join(tokens))
    if len(tokens) > 0:
        return tokens[0]
    return None

def eat(tokens):
    print("eat:" + tokens[0])
    tokens.pop(0)
    
def consume(tok, tokens):
    if tok == peek(tokens):
        eat(tokens)
    else:
        error()
    
def error():
    raise ValueError("illegal expression")

def e(tokens):
    term = t(tokens)
    print("current term:" + repr(term))
    tmp = peek(tokens)
    if tmp == None:
        return term
    while tmp in ('+', '-'):
        eat(tokens)
        tree = []
        tree.append(tmp)
        tree.append(term)
        expr = e(tokens)
        tree.append(expr)
        return tree
    else:
        return term
        
def t(tokens):
    factor = f(tokens)
    tmp = peek(tokens)
    if tmp == None:
        return factor
    while tmp in ('*', '/'):
        eat(tokens)
        term = t(tokens)
        return [tmp, factor, term]
    else:
        return factor
        
def f(tokens):
    tmp = peek(tokens)
    if tmp == '(':
        eat(tokens)
        subtree = e(tokens)
        consume(')', tokens)
        return subtree
    elif tmp.isdigit():
        eat(tokens)
        return tmp
    else:
        error()

def parse(tokens):
#    try:
        tree = e(tokens)
        if len(tokens) > 0:
            error()
        else:
            print("tree:" + repr(tree))
#    except Exception as error:
#        print ("exception: " + repr(error))
    
def repl():
    while True:
        parse(tokenize(input("expression>")))

repl()


def selftest():
    exprs = ("5+3", "55", "(8+9)*3", "-5", "7++7", "(8)", "()")
    for e in exprs:
        print(e)
        parse(tokenize(e))
        
#selftest()
