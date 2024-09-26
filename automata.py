from __future__ import annotations
from collections import defaultdict
from re import RegexFlag

# credit to Computerphile: https://www.youtube.com/watch?v=32bC33nJR3A

class FA:
    def __init__(self,Q:set,Sigma:set,delta,q0,F):
        self.Q = Q # set of states
        self.Sigma = Sigma # set of symbols
        self.delta = delta # transition function as a nested dictionary
        self.q0 = q0 # initial state
        self.F = F # set of final states

    def __repr__(self):
        return f"FA({self.Q},\n\t{self.Sigma},\n\t{self.delta},\n\t{self.q0},\n\t{self.F})"

    def run(self,input):
        pass

class DFA(FA):
    def __init__(self,Q,Sigma,delta,q0,F):
        super(DFA, self).__init__(Q,Sigma,delta,q0,F)

    def __repr__(self):
        return "D" + super(DFA,self).__repr__()

    def run(self,w):
        q = self.q0
        while w != "":
            q = self.delta[q][w[0]]
            w = w[1:]
        return q in self.F

    __call__ = run

    def to_tikz(self):
        heading = f"%move this before document\n\\usepackage{{tikz}}\n\\usetikzlibrary{{arrows.meta,automata,positioning}}\n"
        state_style = "thick,minimum size=1cm"
        begin = f"%DFA\n\\begin{{tikzpicture}}\n\t[shorten >=1pt,node distance=2cm,on grid,>={{Stealth[round]}},\n\tinitial text=start,\n\tevery state/.style={{{state_style}}}]\n\n"
        nodes = "\n".join(
                    ["\t" + f"\\node[state{(',initial' if state == self.q0 else '') + (',accepting' if state in self.F else '')}]".ljust(31)
                     + f"({state})".ljust(4) + f"{'' if state == self.q0 else '[right=of  ]'}".ljust(13)
                     + f"{{${state}$}};" for state in self.Q]) \
                + '\n\n'
        edges = "\t\\path[->]\n" + \
                "\n".join(
                    [f"\t\t({state})\n" + \
                     "\n".join(
                         ["\t\tedge " + \
                          f"{'[loop above]' if self.delta[state][symbol] == state else '[          ]'}".ljust(13) + \
                          f"node {'[above]' if self.delta[state][symbol] != state else ''}".ljust(13) + \
                          f"{{{symbol}}} ({self.delta[state][symbol] if self.delta[state][symbol] != state else ''})"
                          for symbol in self.delta[state]])
                     for state in self.Q]) + ';\n'
        end = "\\end{tikzpicture}"
        return heading + begin + nodes + edges + end

D0 = DFA(Q={0,1,2},
         Sigma={"a","b"},
         delta={
             0:{"a":0, "b":1},
             1:{"a":2, "b":1},
             2:{"a":2, "b":2}},
         q0=0,
         F={0,1})

D1 = DFA(Q={0,1,2,3},
         Sigma={"a","b"},
         delta={
             0:{"a":1, "b":2},
             1:{"a":0, "b":3},
             2:{"a":3, "b":0},
             3:{"a":2, "b":1}},
         q0=0,
         F={0,3})

# print(D1.to_tikz())


# credit to Computerphile: https://www.youtube.com/watch?v=NhWDVqR4tZc
class NFA(FA):
    def __init__(self,Q,Sigma,delta,q0,F):
        super(NFA, self).__init__(Q,Sigma,delta,q0,F)
        self.delta = defaultdict(lambda: defaultdict(set))
        for q in delta:
            for w in delta[q]:
                self.delta[q][w] = set(delta[q][w])

        self.epsilon = ""
        self.epsilon_closure = defaultdict(set)
        self.compute_epsilon_closure()

    def __repr__(self) :
        return "N" + super(NFA, self).__repr__()

    def compute_epsilon_closure(self):
        while True:
            changed = False
            for q in self.Q:
                old = self.epsilon_closure[q]
                self.epsilon_closure[q] |= self.delta[q][self.epsilon]
                if self.epsilon_closure[q] != old:
                    changed = True
            if not changed:
                break

    def run(self,w):
        P = {self.q0}
        print('initial:',P)
        while True:
            # if there are epsilon transitions, take them immediately
            Pnew = set()
            for q in P:
                Pnew |= self.epsilon_closure[q]
            if Pnew != set():
                P |= Pnew

            print(P)
            if w == "":
                break

            # get symbol and take transition
            Pnew = set()
            for q in P:
                Pnew |= self.delta[q][w[0]]

            w = w[1:]
            P = Pnew

        return (P & self.F) != set()

    def __call__(self,x):
        return self.run(x)

    def DFA(self):
        pass


N0 = NFA(Q={0,1,2},
         Sigma={"0","1"},
         delta={
             0: {"0": {0}, "1": {0,1}},
             1: {"0": {2}, "1": {2}}},
         q0=0,
         F={2})

N1 = NFA(Q={0,1,2,3,4},
         Sigma={"0","1"},
         delta={
             0: {"": {1,3}},
             1: {"0": {2}, "1": {1}},
             2: {"0": {1}, "1": {2}},
             3: {"1": {4}, "0": {3}},
             4: {"1": {3}, "0": {4}}},
         q0=0,
         F={1,3})

print(N1("110001"))

from dataclasses import dataclass

@dataclass
class RE:
    def simplify(self) -> RE:
        return self
    pass


class Phi(RE):
    def __str__(self):
        return "(phi)"


class Epsilon(RE):
    def __str__(self):
        return "(epsilon)"


@dataclass
class Symbol(RE):
    symbol: str
    def __str__(self):
        return self.symbol


@dataclass
class Union(RE):
    res: list[RE]
    def __str__(self):
        return "(" + " union ".join(map(str,self.res)) + ")"

    def simplify(self):
        new_res = []
        for re in self.res:
            match re:
                case Phi():
                    continue
                case _:
                    new_res.append(re.simplify())
        if len(new_res) == 0:
            return Phi()
        if len(new_res) == 1:
            return new_res[0].simplify()
        return Union(new_res)

@dataclass
class Concat(RE):
    res: list[RE]
    def __str__(self):
        return "(" + "".join(map(str,self.res)) + ")"

    def simplify(self):
        new_res = []
        for re in self.res:
            if isinstance(re, Phi):
                return Phi()
            elif isinstance(re, Epsilon):
                continue
            else:
                new_res.append(re.simplify())
        if len(new_res) == 0:
            return Epsilon()
        if len(new_res) == 1:
            return new_res[0].simplify()
        return Concat(new_res)

@dataclass
class Star(RE):
    re: RE
    def __str__(self):
        return str(self.re) + "*"

    def simplify(self):
        reg = self.re.simplify()
        if isinstance(reg, Phi) or isinstance(reg, Epsilon):
            return Epsilon()
        else:
            return Star(reg)

def dfa_to_re(dfa: DFA):
    # DFA to GNFA

    states = list(dfa.Q)

    new_start = "new_start"
    new_accept = "new_accept"

    augmented_states = [new_start] + states + [new_accept]

    # gnfa = {}
    # for s1,d2 in dfa.delta.items():
    #     for a,s2 in d2.items():
    #         re = Symbol(str(a))
    #         gnfa[(s1,s2)] = Symbol(str(a))
    gnfa: dict = {(s1,s2):Symbol(str(a)) for s1,d2 in dfa.delta.items() for a,s2 in d2.items()}

    for (s1,s2),re in gnfa.items():
        print(s1, s2, re)

    for s1 in augmented_states[:-1]:
        for s2 in augmented_states[1:]:
            if (s1,s2) not in gnfa:
                if s1 == new_start and s2 == dfa.q0:
                    gnfa[(s1,s2)] = Epsilon()
                elif s1 in dfa.F and s2 == new_accept:
                    gnfa[(s1,s2)] = Epsilon()
                else:
                    gnfa[(s1,s2)] = Phi()

    for s1 in augmented_states:
        for s2 in augmented_states:
            if (s1,s2) in gnfa:
                print(s1,s2,gnfa[(s1,s2)])

    # GNFA to RE
    while len(states) > 0:
        remove_state = states.pop(0)
        for s1 in [new_start] + states:
            for s2 in states + [new_accept]:
                gnfa[(s1,s2)] = Union([
                    gnfa[(s1,s2)],
                    Concat([
                        gnfa[(s1,remove_state)],
                        Star(gnfa[(remove_state,remove_state)]),
                        gnfa[(remove_state,s2)],
                    ]),
                ]).simplify()

        # for s1 in augmented_states:
        #     for s2 in augmented_states:
        #         if (s1,s2) in gnfa:
        #             print(s1,s2,gnfa[(s1,s2)])

    print(gnfa)
    print(gnfa[new_start,new_accept].simplify())

M = DFA(
    Q={0,1,2,3},
    Sigma={"a","b"},
    delta={
        0: {"a": 1, "b": 2},
        1: {"a": 0, "b": 3},
        2: {"a": 3, "b": 0},
        3: {"a": 2, "b": 1},
    },
    q0=0,
    F={3},
)

dfa_to_re(M)

from itertools import product,combinations

def distinguishable(dfa: DFA):
    alphabet = dfa.Sigma

    non_accept = list(dfa.Q - dfa.F)
    accept = list(dfa.F)

    distinguishable = set(product(non_accept, accept)) | set(product(accept, non_accept))

    not_yet_distinguished = set(product(dfa.Q, dfa.Q)) - distinguishable

    while True:
        to_add = set()
        for pair in not_yet_distinguished:
            s1,s2 = pair
            for symbol in alphabet:
                if (dfa.delta[s1][symbol], dfa.delta[s2][symbol]) in distinguishable:
                    to_add.add((s1,s2))
        if len(to_add) == 0:
            break
        not_yet_distinguished -= to_add
        distinguishable |= to_add

    return distinguishable, not_yet_distinguished

M2 = DFA(
    Q={0,1,2,3},
    Sigma={"0","1"},
    delta={
        0: {"0": 2, "1": 1},
        1: {"0": 3, "1": 0},
        2: {"0": 3, "1": 1},
        3: {"0": 2, "1": 0},
    },
    q0=0,
    F={2,3},
)

d,nd = distinguishable(M2)

print(d)

print(nd)
