# from functools import lru_cache
# import sys
# sys.setrecursionlimit(10**9)
# # input = sys.stdin.readline
# from decimal import Decimal
# from functools import cmp_to_key
# from collections import Counter
# from itertools import permutations
# from itertools import combinations
# from itertools import combinations_with_replacement
# from itertools import product
# from itertools import accumulate
# from itertools import groupby
# from itertools import pairwise
# from copy import deepcopy
# import networkx as nx
# import networkx.algorithms as nxa
# import numpy as np
# import math
import heapq
# from collections import OrderedDict
# import bisect
# from collections import deque
from collections import defaultdict

def main():
    N,M=map(int,input().split())
    A=list(range(N))
    event=[]
    eat=[0 for _ in range(N)]
    for i in range(M):
        T,W,S=map(int,input().split())
        event.append((T,W,S))
        
    while event:
        T,W,S = heapq.heappop(event)
        if W == -1:
            heapq.heappush(A,S)
        else:
            if len(A) == 0:
                continue
            eater=heapq.heappop(A)
            eat[eater]+=W
            #Sの部分に人を記録しておくと取り出したときに並び直す人を一発で特定できる
            heapq.heappush(event,(T+S,-1,eater))
    for eater in eat:
        print(eater)


class UnionFind():
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x):
        return -self.parents[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def group_count(self):
        return len(self.roots())

    def all_group_members(self):
        group_members = defaultdict(list)
        for member in range(self.n):
            group_members[self.find(member)].append(member)
        return group_members

    def __str__(self):
        return '\n'.join(f'{r}: {m}' for r, m in self.all_group_members().items())
def yes():
    print("Yes")
def no():
    print("No")
def minusOne():
    print(-1)
    
if __name__ == "__main__":
    main()