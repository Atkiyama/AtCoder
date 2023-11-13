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
INF = 10 ** 18
dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]
dxy = [(1, 0), (0, 1), (-1, 0), (0, -1)]

zeroset=set([])
oneset=set([])
def main():
    N,M=map(int, input().split())
    A=list(map(int, input().split()))
    B=list(map(int, input().split()))
    
    for i in range(M):
        print("Zeroset:", zeroset)  # セットが空であることを示す出力
        print("Oneset:", oneset)    # セットが空であることを示す出力
        check(A,B,i)
        
                
        
    yes()

def check(A,B,i):
    A0 =A[i] in zeroset
    A1=A[i] in oneset
    B0=B[i] in zeroset
    B1=B[i] in oneset
    
    if (A0 and B0) or (A1 and B1) or (A[i]==B[i]):
        no()
        exit()
    else:
        if A0 and not B0 and not B1:
            oneset.add(B[i])
        elif A1 and not B0 and not B1:
            zeroset.add(B[i])
        elif B0 and not A0 and not A1:
            oneset.add(A[i])
        elif B1 and not A0 and not A1:
            zeroset.add(A[i])
        else:
            if A[i]%2==0:
                zeroset.add(A[i])
                oneset.add(B[i])
            else:
                zeroset.add(B[i])
                oneset.add(A[i])
    

            
            
    
def judge(A,n):
    if n in A and n-1 in A and n+1 in A:
        return True
    else:
        return False


def swap(A, i, j):
    tmp = A[i]
    A[i] = A[j]
    A[j] = tmp

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
