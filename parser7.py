'''
version 0:
E <- E (+|-|*|/) E | (E) | NUM

version 1:
E <- T { +/- E}
T <- F { *// T}
F <- NUM | (E)

version 2: add uniary
E <- T { +/- E}
T <- F { *// T}
F <- NUM | (E) | -T //note -T not -E -5+3

version 3: add power
E <- T { +/- T} //note E to T to solve left associative -
T <- P { *// P} //note T to P to solve left associative /
P <- F { ^ P }
F <- NUM | (E) | -T
'''

import re

#tokenizer
def tokenize(line):
    return [_f for _f in re.split(r'(\+|-|\*|/|\(|\)|\^)', "".join(line.split())) if _f]

#print tokenize("5^2")

#parser utility functions
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

#parser recursive functions
def e(tokens):
    term = t(tokens)
    tmp = peek(tokens)
    while tmp in ('+', '-'):
        eat(tokens)
        term = [tmp, term, t(tokens)]
        tmp = peek(tokens)
    return term
 
def t(tokens):
    power = p(tokens)
    tmp = peek(tokens)
    while tmp in ('*', '/'):
        eat(tokens)
        power = [tmp, power, p(tokens)]
        tmp = peek(tokens)
    return power

def p(tokens):
    power = f(tokens)
    tmp = peek(tokens)
    while tmp == '^':
        eat(tokens)
        return [tmp, power, p(tokens)]
    else:
        return power
        
def f(tokens):
    tmp = peek(tokens)
    if tmp == '(':
        eat(tokens)
        subtree = e(tokens)
        consume(')', tokens)
        return subtree
    elif tmp == '-':
        eat(tokens)
        return ['-', t(tokens)]
    elif tmp.isdigit():
        eat(tokens)
        return int(tmp)
    else:
        error()

def parse(tokens):
    try:
        tree = e(tokens)
        if len(tokens) > 0:
            error()
        else:
            return tree
    except Exception as error:
        print(("exception: " + repr(error)))

#print parse(tokenize("4+3+2+1"))

def plus(x, y):
    return x + y

def minus(x, y):
    return x - y

def times(x, y):
    return x * y

def divides(x, y):
    return x / y
    
def power(x, y):
    return x ** y
    
switcher = {
    '+': plus,
    '-': minus,
    '*': times,
    '/': divides,
    '^': power,
}

def uniary(tree):
    return len(tree) == 2
    
def evaluate(tree):
    if type(tree) is not list:
        return tree
    elif tree[0] in "+-*/^":
        if uniary(tree):
            return -evaluate(tree[1])
        else:
            return switcher.get(tree[0])(evaluate(tree[1]), evaluate(tree[2]))
    else:
        print("invalid tree")
        
#interpreter repl
def repl():
    while True:
        print(evaluate(parse(tokenize(input("expression>")))))

repl()

#self test cases 
def selftest():
    exprs = ("5+3", "55", "(8+9)*3", "-5", "7++7", "(8)", "()", "(3+4)*(8-7)", "3-2-1", "4/2/2")
    for e in exprs:
        print(e, end=' ')
        print(evaluate(parse(tokenize(e))))
        
#selftest()
