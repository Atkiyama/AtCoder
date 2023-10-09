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
# import heapq
# from collections import OrderedDict
# import bisect
# from collections import deque
from collections import defaultdict

'''
https://twitter.com/e869120/status/1388262816101007363/photo/1
二次元配列を作成してそれに値を加算していくことで解がもとまる
このときに二次元いもす法が使えるのでそれを覚えよう
'''

def main():
        # Input
    N = int(input())
    lx = [0] * (1 << 18)
    ly = [0] * (1 << 18)
    rx = [0] * (1 << 18)
    ry = [0] * (1 << 18)

    # Count Number
    cnt = [[0] * 1009 for _ in range(1009)]
    Answer = [0] * (1 << 18)

    # Step #1. Input
    for i in range(1, N + 1):
        lx[i], ly[i], rx[i], ry[i] = map(int, input().split())

    # Step #2. Imos Method in 2D
    for i in range(1, N + 1):
        cnt[lx[i]][ly[i]] += 1
        cnt[lx[i]][ry[i]] -= 1
        cnt[rx[i]][ly[i]] -= 1
        cnt[rx[i]][ry[i]] += 1

    for i in range(1001):
        for j in range(1, 1001):
            cnt[i][j] += cnt[i][j - 1]

    for i in range(1, 1001):
        for j in range(1001):
            cnt[i][j] += cnt[i - 1][j]

    # Step #3. Count Number
    for i in range(1001):
        for j in range(1001):
            if cnt[i][j] >= 1:
                Answer[cnt[i][j]] += 1

    # Step #4. Output The Answer
    for i in range(1, N + 1):
        print(Answer[i])


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