
from collections import deque, defaultdict

input = [("a","b",2),("b","c",3)]

graph = defaultdict(list)

# convert input into structure
for src,dst,weight in input:
    graph[src].append((dst,weight))
    graph[dst].append((src,1/weight))

# actual bfs
def bfs(graph,src,dst):
    visited = set()
    queue = deque([(src, 1.0)])
    if not graph[src] or not graph[dst]:
        return -1 # either node is not in the graph.
    while queue:
        curr, multiplier = queue.popleft()
        if curr == dst:
            return multiplier
        visited.add(curr)
        for next, weight in graph[curr]:
            if next in visited:
                continue
            queue.append((next, multiplier * weight))

    return -1

queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
# print(bfs(graph,"a","c"))
for query in queries:
    print(bfs(graph, query[0], query[1]))

