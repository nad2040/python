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
    t(tokens)
    while peek(tokens) in ('+', '-'):
        eat(tokens)
        e(tokens)
        
def t(tokens):
    f(tokens)
    while peek(tokens) in ('*', '/'):
        eat(tokens)
        t(tokens)
        
def f(tokens):
    tmp = peek(tokens)
    if tmp == '(':
        eat(tokens)
        e(tokens)
        consume(')', tokens)
    elif tmp.isdigit():
        eat(tokens)
    else:
        error()

def parse(tokens):
    try:
        e(tokens)
        if len(tokens) > 0:
            error()
        else:
            print("valid expression")
    except Exception as error:
        print(("exception: " + repr(error)))
    
def repl():
    while True:
        parse(tokenize(input("expression>")))

repl()