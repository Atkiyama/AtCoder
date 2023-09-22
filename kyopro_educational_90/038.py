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
import math
# import heapq
# from collections import OrderedDict
# import bisect
# from collections import deque
from collections import defaultdict
'''
https://twitter.com/e869120/status/1392612322410057729
オーバーフローに気をつける問題だがpythonでint型の場合は問題にならない
'''
def main():
    A,B=map(int,input().split())
    ans=math.lcm(A,B)
    if ans >10 ** 18:
        print("Large")
    else:
        print(math.lcm(A,B))
        
        


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