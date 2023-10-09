# from functools import lru_cache
from collections import defaultdict
from itertools import product
import sys
sys.setrecursionlimit(10**9)
# # input = sys.stdin.readline
# from decimal import Decimal
# from functools import cmp_to_key
# from collections import Counter
# from itertools import permutations
# from itertools import combinations
# from itertools import combinations_with_replacement
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
INF = 10 ** 18
dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]
dxy = [(1, 0), (0, 1), (-1, 0), (0, -1)]

'''
三つのパネルを再起的に重ねる
'''


def main():
    P1 = []
    P2 = []
    P3 = []
    for i in range(4):
        P1.append(input())
    for i in range(4):
        P2.append(input())
    for i in range(4):
        P3.append(input())

    p1 = change(P1)
    p2 = change(P2)
    p3 = change(P3)
    ps = [p1, p2, p3]

    for _ in range(4):
        for _ in range(4):
            dfs(0, [[0] * 4 for _ in range(4)], ps)
            ps[2] = rotate(ps[2])
        ps[1] = rotate(ps[1])
    no()


'''
i:現在配置してるポリオミノの番号
pannel:パネルの土台
ps:ポリオミノの配列
'''


def dfs(i, pannel, ps):
    if i == 3:

        for j in range(4):
            for k in range(4):
                if pannel[j][k] != 1:
                    return
        yes()
        exit(0)
    # if i == 3:
    #     if all(all(cell == 1 for cell in row) for row in pannel):
    #         yes()
        return
    for di in range(-3, 10):
        for dj in range(-3, 10):
            ex2 = [row[:] for row in pannel]
            if can_put(ex2, ps[i], di, dj):
                dfs(i+1, ex2, ps)

# ポリオミノを置くメソッド
# ちゃんとミノがおけるならtrueを返す


def can_put(ex2, p, di, dj):
    for i in range(4):
        for j in range(4):
            if p[i][j] == 1:
                ni = i+di
                nj = j+dj
                if (not checkIndex2(ex2, ni, nj)) or ex2[ni][nj] == 1:
                    return False
                ex2[ni][nj] = 1

    return True


# ポリオミノを右に90度回転
def rotate(matrix):
    N = len(matrix)
    M = len(matrix[0])
    rotated = [[0] * N for _ in range(M)]
    for i in range(N):
        for j in range(M):
            rotated[j][N-1-i] = matrix[i][j]
    return rotated


# # と .で表現された二次元配列を01で表現する
def change(P):
    p = []
    for i in P:
        line = []
        for j in i:
            if j == '#':
                line.append(1)
            else:
                line.append(0)
        p.append(line)
    return p


def swap(A, i, j):
    tmp = A[i]
    A[i] = A[j]
    A[j] = tmp


def basezero(num):
    return num-1


def checkIndex(list, i):
    length = len(list)

    if 0 <= i < length:
        return True
    else:
        return False


def checkIndex2(list, i, j):
    H = len(list)
    W = len(list[0])
    if 0 <= i < H and 0 <= j < W:
        return True
    else:
        return False


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
