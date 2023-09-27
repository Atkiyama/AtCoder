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
from collections import deque
from collections import defaultdict
INF = 10 ** 18
dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]
dxy=[(1,0),(0,1),(-1,0),(0,-1)]
'''
方向関係の実装がうまくいかないので後でやろう
'''
def main():
    H,W=map(int,input().split())
    rs,cs = map(int,input().split())
    rt,ct = map(int,input().split())
    rs,cs = map(basezero,(rs,cs))
    rt,ct = map(basezero,(rt,ct))
    S=[]
    

    for i in range(H):
        S.append(list(input()))
    
    
    print(bfs(H,W,rs,cs,rt,ct,S))

def bfs(H,W,rs,cs,rt,ct,S):
    q=deque([(rs,cs)])
    dist=[[[INF for _ in range(4)] for _ in range(W)] for _ in range(H)]
    for i in range(4):
        dist[rs][cs][i]=0
    history=[[ False for _ in range(W)] for _ in range(H)]
    history[rs][cs]=True
    for i in range(H):
        for j in range(W):
            if S[i][j] =='#':
                history[i][j]=True
    print(history)
    while q:
        now = q.pop()
        print(now)
        for i in range(4):
            x=now[0]+dx[i]
            y=now[1]+dy[i]
            
            if checkIndex2(history,x,y) and not history[x][y]:
                q.append((x,y))
                history[x][y]=True
                for j in range(4):
                    if i!=j:
                        next=dist[now[0]][now[1]][j]+1
                    else:
                        next=dist[now[0]][now[1]][j]
                    if next < dist[x][y][j]:
                        dist[x][y][j]=next
    sum=0
    for i in range(4):
        sum+=dist[rt][ct][i]
    
        
    
    return sum 
def checkIndex2(list,i,j):
    H=len(list)
    W=len(list[0])
    if 0<=i<H and 0<=j<W:
        return True
    else:
        return False

def basezero(num):
    return num-1

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