#!/usr/bin/env python3

from collections import defaultdict

class MaxFlow:
    def __init__(self, graph: dict[tuple[str,str], int], source, sink):
        self.graph = defaultdict(int)
        self.flow = defaultdict(int)
        for a,b in graph:
            self.graph[a,b] = graph[a,b]
            self.graph[b,a] = 0 # must initialize backwards capacity as 0 to allow undo/return of flow
        self.source = source
        self.sink = sink
        self.path = [self.source]

    def solve(self):
        self._solve()
        flow = {}
        for a,b in self.graph:
            if self.graph[a,b] > 0:
                flow[a,b] = self.flow[a,b]
        print(flow)
        print(sum(self.flow[a,b] for a,b in self.flow if a == self.source))

    def _solve(self):
        # Ford-Fulkerson Method w/ DFS
        current = self.path[-1]
        if current == self.sink:
            print(self.path)
            rates = [(self.graph[a,b] - self.flow[a,b]) for a,b in zip(self.path[:-1],self.path[1:])]
            bottleneck_rate = min(rates)
            for a,b in zip(self.path[:-1],self.path[1:]):
                self.flow[a,b] += bottleneck_rate
                self.flow[b,a] -= bottleneck_rate
            return

        for u,v in self.graph: # neighbors
            if u == current and v not in self.path: # because we can go backwards, don't loop
                self.path.append(v)
                self._solve()
                self.path.pop()

mf = MaxFlow(
    graph={
        ('a','b'): 1000,
        ('a','c'): 1000,
        ('b','c'): 1,
        ('b','d'): 1000,
        ('c','d'): 1000
    },
    source='a',
    sink='d'
)

mf.solve()

