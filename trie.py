class Trie:
    def __init__(self):
        self.children = {}

    def insert(self, word: str) -> None:
        parent = self.children
        for c in word:
            d = parent.get(c, {})
            parent[c] = d
            parent = parent[c]
        parent[""] = {}

    def search(self, word: str) -> bool:
        trav = self.children
        for c in word:
            if trav == {}:
                return False
            if c not in trav:
                return False
            trav = trav[c]
        return "" in trav

    def startsWith(self, prefix: str) -> bool:
        trav = self.children
        for c in prefix:
            if trav == {}:
                return False
            if c not in trav:
                return False
            trav = trav[c]
        return True
