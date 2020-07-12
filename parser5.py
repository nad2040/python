import re

def tokenize(line):
    return [_f for _f in re.split(r'(\+|-|\*|/|\(|\))', "".join(line.split())) if _f]

def peek(tokens):
    if len(tokens) > 0:
        return tokens[0]
    return None

def eat(tokens):
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
    tmp = peek(tokens)
    if tmp == None:
        return term
    while tmp in ('+', '-'):
        eat(tokens)
        return [tmp, term, e(tokens)]
    else:
        return term
        
def t(tokens):
    factor = f(tokens)
    tmp = peek(tokens)
    if tmp == None:
        return factor
    while tmp in ('*', '/'):
        eat(tokens)
        return [tmp, factor, t(tokens)]
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
    try:
        tree = e(tokens)
        if len(tokens) > 0:
            error()
        else:
            print("tree:" + repr(tree))
    except Exception as error:
        print(("exception: " + repr(error)))
    
def repl():
    while True:
        parse(tokenize(input("expression>")))

#repl()


def selftest():
    exprs = ("5+3", "55", "(8+9)*3", "-5", "7++7", "(8)", "()", "(3+4)*(8-7)")
    for e in exprs:
        print(e, end=' ')
        parse(tokenize(e))
        
selftest()
