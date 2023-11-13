from functools import lru_cache
import sys
sys.setrecursionlimit(10**9)
# input = sys.stdin.readline
from decimal import Decimal
from functools import cmp_to_key
from collections import Counter
from itertools import permutations
from itertools import combinations
from itertools import combinations_with_replacement
from itertools import product
from itertools import accumulate
from itertools import groupby
from itertools import pairwise
from copy import deepcopy
import networkx as nx
import networkx.algorithms as nxa
import numpy as np
import math
import heapq
from collections import OrderedDict
import bisect
from collections import deque
from collections import defaultdict
'''
https://twitter.com/e869120/status/1387538790017769474/photo/1
二部グラフの性質を使う問題
二部グラフ:隣接するグラフどうしで違う色に必ず塗り分けられるグラフのこと
木は必ず二部グラフになるのでこの性質が使える
'''

def main():
    N=int(input())
    graph=defaultdict(list)
    color=[False for _ in range(N)]
    for i in range(N-1):
        A,B=map(int,input().split())
        A-=1
        B-=1
        graph[A].append(B)
        graph[B].append(A)
    
    color= dfs(graph,0,N)

    color1=[]
    color2=[]
    for i in range(len(color)):
        if color[i]:
            color1.append(i+1)
        else:
            color2.append(i+1)
    #print(color)
    if len(color1)>len(color2):
        color1.sort()
        for i in range(N//2):
            print(str(color1[i]),end=" ")
    else:
        color2.sort()
        for i in range(N//2):
            print(str(color2[i]),end=" ")
    #print(color)
    
def dfs(graph,start,N):
    q = deque([start])
    color=[False for _ in range(N)]
    history=set()
    history.add(start)
    color[start]=True
    while q:
        now= q.pop()
        for near in graph[now]:
            if not near in history:
                #print(near+1)
                if color[now]:
                    color[near]=False
                else:
                    color[near]=True
                q.append(near)
                history.add(near)
        
    return color
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