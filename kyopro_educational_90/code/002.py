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
https://twitter.com/e869120/status/1377027868518064129
ビット全探索を使う問題
Nが少ないのでN^2でいける時に使う
()みたいに二択の連続の場合に使う
'''
def main():
    
    N=int(input())
    for i in range(1 <<N):
        st=""
        for j in range(N-1,-1,-1):
            if(i & (1 <<j) ==0):
                st+="("
            else:
                st+=")"
        if judge(st):
            print(st)
        
def judge(st):
    front=0
    back=0
    for i in st:
        if i=="(":
            front+=1
        else:
            back+=1
        if front<back:
            return False
    if front==back:
        return True
    else:
        return False
        
    
        
    


if __name__ == "__main__":
    main()