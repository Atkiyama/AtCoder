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
https://twitter.com/e869120/status/1378115289649348611
そのままやるとTLEになるので前処理を行っておくことで計算量を減らす問題
今回の問題では縦列と横列の合計をあらかじめだしておく
'''
def main():
    
    H,W=map(int,input().split())
    A = [list(map(int, input().split())) for _ in range(H)]
    hSum=[]
    wSum=[]
    for i in range(H):
        hSum.append(0)
        for j in range(W):
            hSum[i]+=A[i][j]
    for i in range(W):
        wSum.append(0)
        for j in range(H):
            wSum[i]+=A[j][i]
    ans=[[0 for _ in range(W)] for _ in range(H)]
    for i in range(H):
        for j in range(W):
            ans[i][j]=hSum[i]+wSum[j]-A[i][j]
            print(ans[i][j],end=" ")
        print()
    
    
    
            
if __name__ == "__main__":
    main()