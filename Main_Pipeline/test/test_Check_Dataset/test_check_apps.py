from datasets import load_dataset
import json

# 加载 default、queries、corpus
cache_dir = "/data/share/project/shared_datasets/mteb"
ds_default = load_dataset("mteb/apps", "default", cache_dir=cache_dir)
ds_queries = load_dataset("mteb/apps", "queries", cache_dir=cache_dir)
ds_corpus = load_dataset("mteb/apps", "corpus", cache_dir=cache_dir)

# 建立 _id -> 整行 的映射，便于按 id 查找
queries_list = ds_queries["queries"]
corpus_list = ds_corpus["corpus"]
query_by_id = {row["_id"]: row for row in queries_list}
corpus_by_id = {row["_id"]: row for row in corpus_list}

# 从 test 中随机抽取 5 条
test_split = ds_default["test"]
sampled_ds = test_split.shuffle(seed=123).select(range(5))
first_five = sampled_ds.to_dict()

print("=" * 80)
print("Default 数据集 test 集合随机 5 条：query 与 corpus 具体内容")
print("=" * 80)

# 用于 Few_Shot_Example.py 兼容格式的列表
apps_few_shot_format = []

for i in range(5):
    query_id = first_five["query-id"][i]
    corpus_id = first_five["corpus-id"][i]
    score = first_five["score"][i]

    query_row = query_by_id[query_id]
    corpus_row = corpus_by_id[corpus_id]

    # 收集 Few_Shot_Example 兼容格式：query + Positive(corpus text)
    apps_few_shot_format.append({
        "query": query_row["text"],
        "Positive": corpus_row["text"]
    })

    print(f"\n【第 {i+1} 条】")
    print(f"  query-id:  {query_id}")
    print(f"  corpus-id: {corpus_id}")
    print(f"  score:     {score}")
    print("-" * 60)
    print("  QUERY 内容:")
    print(f"    {query_row['text']}")
    print("-" * 60)
    print("  CORPUS 内容:")
    print(f"    title: {corpus_row.get('title', '')}")
    text = corpus_row["text"]
    print(f"    text:  {text}")
    print("=" * 80)

# 输出 Few_Shot_Example.py 中 "apps" 兼容的 Python 结构（可直接复制到 Few_Shot_Example.py）
print("\n" + "=" * 80)
print("Few_Shot_Example 兼容格式（可复制到 Few_Shot_Example.py 的 \"apps\" 键）：")
print("=" * 80)
print('    "apps": ', end="")
print(json.dumps(apps_few_shot_format, indent=4, ensure_ascii=False))
print("=" * 80)


'''
================================================================================
Few_Shot_Example 兼容格式（可复制到 Few_Shot_Example.py 的 "apps" 键）：
================================================================================
    "apps": [
    {
        "query": "A tree is an undirected connected graph without cycles. The distance between two vertices is the number of edges in a simple path between them.\n\nLimak is a little polar bear. He lives in a tree that consists of n vertices, numbered 1 through n.\n\nLimak recently learned how to jump. He can jump from a vertex to any vertex within distance at most k.\n\nFor a pair of vertices (s, t) we define f(s, t) as the minimum number of jumps Limak needs to get from s to t. Your task is to find the sum of f(s, t) over all pairs of vertices (s, t) such that s < t.\n\n\n-----Input-----\n\nThe first line of the input contains two integers n and k (2 ≤ n ≤ 200 000, 1 ≤ k ≤ 5) — the number of vertices in the tree and the maximum allowed jump distance respectively.\n\nThe next n - 1 lines describe edges in the tree. The i-th of those lines contains two integers a_{i} and b_{i} (1 ≤ a_{i}, b_{i} ≤ n) — the indices on vertices connected with i-th edge.\n\nIt's guaranteed that the given edges form a tree.\n\n\n-----Output-----\n\nPrint one integer, denoting the sum of f(s, t) over all pairs of vertices (s, t) such that s < t.\n\n\n-----Examples-----\nInput\n6 2\n1 2\n1 3\n2 4\n2 5\n4 6\n\nOutput\n20\n\nInput\n13 3\n1 2\n3 2\n4 2\n5 2\n3 6\n10 6\n6 7\n6 13\n5 8\n5 9\n9 11\n11 12\n\nOutput\n114\n\nInput\n3 5\n2 1\n3 1\n\nOutput\n3\n\n\n\n-----Note-----\n\nIn the first sample, the given tree has 6 vertices and it's displayed on the drawing below. Limak can jump to any vertex within distance at most 2. For example, from the vertex 5 he can jump to any of vertices: 1, 2 and 4 (well, he can also jump to the vertex 5 itself). [Image] \n\nThere are $\\frac{n \\cdot(n - 1)}{2} = 15$ pairs of vertices (s, t) such that s < t. For 5 of those pairs Limak would need two jumps: (1, 6), (3, 4), (3, 5), (3, 6), (5, 6). For other 10 pairs one jump is enough. So, the answer is 5·2 + 10·1 = 20.\n\nIn the third sample, Limak can jump between every two vertices directly. There are 3 pairs of vertices (s < t), so the answer is 3·1 = 3.",
        "Positive": "\"\"\"\n#If FastIO not needed, used this and don't forget to strip\n#import sys, math\n#input = sys.stdin.readline\n\"\"\"\n\nimport os\nimport sys\nfrom io import BytesIO, IOBase\nimport heapq as h \nfrom bisect import bisect_left, bisect_right\n\nfrom types import GeneratorType\nBUFSIZE = 8192\nclass FastIO(IOBase):\n    newlines = 0\n \n    def __init__(self, file):\n        import os\n        self.os = os\n        self._fd = file.fileno()\n        self.buffer = BytesIO()\n        self.writable = \"x\" in file.mode or \"r\" not in file.mode\n        self.write = self.buffer.write if self.writable else None\n \n    def read(self):\n        while True:\n            b = self.os.read(self._fd, max(self.os.fstat(self._fd).st_size, BUFSIZE))\n            if not b:\n                break\n            ptr = self.buffer.tell()\n            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)\n        self.newlines = 0\n        return self.buffer.read()\n \n    def readline(self):\n        while self.newlines == 0:\n            b = self.os.read(self._fd, max(self.os.fstat(self._fd).st_size, BUFSIZE))\n            self.newlines = b.count(b\"\\n\") + (not b)\n            ptr = self.buffer.tell()\n            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)\n        self.newlines -= 1\n        return self.buffer.readline()\n \n    def flush(self):\n        if self.writable:\n            self.os.write(self._fd, self.buffer.getvalue())\n            self.buffer.truncate(0), self.buffer.seek(0)\n \n \nclass IOWrapper(IOBase):\n    def __init__(self, file):\n        self.buffer = FastIO(file)\n        self.flush = self.buffer.flush\n        self.writable = self.buffer.writable\n        self.write = lambda s: self.buffer.write(s.encode(\"ascii\"))\n        self.read = lambda: self.buffer.read().decode(\"ascii\")\n        self.readline = lambda: self.buffer.readline().decode(\"ascii\")\n \n \nsys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)\ninput = lambda: sys.stdin.readline().rstrip(\"\\r\\n\")\n\nfrom collections import defaultdict as dd, deque as dq, Counter as dc\nimport math, string\n\n\ndef getInts():\n    return [int(s) for s in input().split()]\n\ndef getInt():\n    return int(input())\n\ndef getStrs():\n    return [s for s in input().split()]\n\ndef getStr():\n    return input()\n\ndef listStr():\n    return list(input())\n\ndef getMat(n):\n    return [getInts() for _ in range(n)]\n\nMOD = 10**9+7\n\n\n\"\"\"\nEach edge goes from parent U to child V\nEdge appears on S_V * (N - S_V) paths\n\nFor each path of length L, (L + (-L)%K)/K\n\n\nL%K 0, 1, 2, 3, 4\n(K - L%K)%K K K-1 K-2 ...\n0 K-1 K-2 ...\n\n\"\"\"\ndef bootstrap(f, stack=[]):\n    def wrappedfunc(*args, **kwargs):\n        if stack:\n            return f(*args, **kwargs)\n        else:\n            to = f(*args, **kwargs)\n            while True:\n                if type(to) is GeneratorType:\n                    stack.append(to)\n                    to = next(to)\n                else:\n                    stack.pop()\n                    if not stack:\n                        break\n                    to = stack[-1].send(to)\n            return to\n    return wrappedfunc\n\ndef solve():\n    N, K = getInts()\n    graph = dd(set)\n    for i in range(N-1):\n        A, B = getInts()\n        graph[A].add(B)\n        graph[B].add(A)\n    dp_count = [[0 for j in range(5)] for i in range(N+1)]\n    dp_total = [0 for j in range(N+1)]\n    nonlocal ans\n    ans = 0\n    @bootstrap\n    def dfs(node,parent,depth):\n        nonlocal ans\n        dp_count[node][depth % K] = 1\n        dp_total[node] = 1\n        for neigh in graph[node]:\n            if neigh != parent:\n                yield dfs(neigh,node,depth+1)\n                for i in range(K):\n                    for j in range(K):\n                        diff = (i+j-2*depth)%K\n                        req = (-diff)%K\n                        ans += req * dp_count[node][i] * dp_count[neigh][j]\n                for i in range(K):\n                    dp_count[node][i] += dp_count[neigh][i]\n                dp_total[node] += dp_total[neigh]\n        ans += dp_total[node] * (N - dp_total[node])\n        yield\n    dfs(1,-1,0)\n    return ans//K\n    \n    \nprint(solve())\n"
    },
    {
        "query": "You are given a rectangular cake, represented as an r × c grid. Each cell either has an evil strawberry, or is empty. For example, a 3 × 4 cake may look as follows: [Image] \n\nThe cakeminator is going to eat the cake! Each time he eats, he chooses a row or a column that does not contain any evil strawberries and contains at least one cake cell that has not been eaten before, and eats all the cake cells there. He may decide to eat any number of times.\n\nPlease output the maximum number of cake cells that the cakeminator can eat.\n\n\n-----Input-----\n\nThe first line contains two integers r and c (2 ≤ r, c ≤ 10), denoting the number of rows and the number of columns of the cake. The next r lines each contains c characters — the j-th character of the i-th line denotes the content of the cell at row i and column j, and is either one of these:   '.' character denotes a cake cell with no evil strawberry;  'S' character denotes a cake cell with an evil strawberry. \n\n\n-----Output-----\n\nOutput the maximum number of cake cells that the cakeminator can eat.\n\n\n-----Examples-----\nInput\n3 4\nS...\n....\n..S.\n\nOutput\n8\n\n\n\n-----Note-----\n\nFor the first example, one possible way to eat the maximum number of cake cells is as follows (perform 3 eats). [Image]  [Image]  [Image]",
        "Positive": "r, c = list(map(int, input().split()))\ncake = [input().strip() for _ in range(r)]\nans = 0\nfor i in range(r):\n    for j in range(c):\n        if cake[i][j] == '.' and ('S' not in cake[i] or 'S' not in list(zip(*cake))[j]):\n            ans += 1\nprint(ans)\n"
    },
    {
        "query": "Every summer Vitya comes to visit his grandmother in the countryside. This summer, he got a huge wart. Every grandma knows that one should treat warts when the moon goes down. Thus, Vitya has to catch the moment when the moon is down.\n\nMoon cycle lasts 30 days. The size of the visible part of the moon (in Vitya's units) for each day is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, and then cycle repeats, thus after the second 1 again goes 0.\n\nAs there is no internet in the countryside, Vitya has been watching the moon for n consecutive days and for each of these days he wrote down the size of the visible part of the moon. Help him find out whether the moon will be up or down next day, or this cannot be determined by the data he has.\n\n\n-----Input-----\n\nThe first line of the input contains a single integer n (1 ≤ n ≤ 92) — the number of consecutive days Vitya was watching the size of the visible part of the moon. \n\nThe second line contains n integers a_{i} (0 ≤ a_{i} ≤ 15) — Vitya's records.\n\nIt's guaranteed that the input data is consistent.\n\n\n-----Output-----\n\nIf Vitya can be sure that the size of visible part of the moon on day n + 1 will be less than the size of the visible part on day n, then print \"DOWN\" at the only line of the output. If he might be sure that the size of the visible part will increase, then print \"UP\". If it's impossible to determine what exactly will happen with the moon, print -1.\n\n\n-----Examples-----\nInput\n5\n3 4 5 6 7\n\nOutput\nUP\n\nInput\n7\n12 13 14 15 14 13 12\n\nOutput\nDOWN\n\nInput\n1\n8\n\nOutput\n-1\n\n\n\n-----Note-----\n\nIn the first sample, the size of the moon on the next day will be equal to 8, thus the answer is \"UP\".\n\nIn the second sample, the size of the moon on the next day will be 11, thus the answer is \"DOWN\".\n\nIn the third sample, there is no way to determine whether the size of the moon on the next day will be 7 or 9, thus the answer is -1.",
        "Positive": "# You lost the game.\n\nn = int(input())\nL = list(map(int, input().split()))\n\nif n == 1:\n    if L[0] == 0:\n        print(\"UP\")\n    elif L[0] == 15:\n        print(\"DOWN\")\n    else:\n        print(\"-1\")\nelse:\n    d = L[n-2] - L[n-1]\n    if d < 0:\n        if L[n-1] == 15:\n            print(\"DOWN\")\n        else:\n            print(\"UP\")\n    else:\n        if L[n-1] == 0:\n            print(\"UP\")\n        else:\n            print(\"DOWN\")\n"
    },
    {
        "query": "Mister B once received a gift: it was a book about aliens, which he started read immediately. This book had c pages.\n\nAt first day Mister B read v_0 pages, but after that he started to speed up. Every day, starting from the second, he read a pages more than on the previous day (at first day he read v_0 pages, at second — v_0 + a pages, at third — v_0 + 2a pages, and so on). But Mister B is just a human, so he physically wasn't able to read more than v_1 pages per day.\n\nAlso, to refresh his memory, every day, starting from the second, Mister B had to reread last l pages he read on the previous day. Mister B finished the book when he read the last page for the first time.\n\nHelp Mister B to calculate how many days he needed to finish the book.\n\n\n-----Input-----\n\nFirst and only line contains five space-separated integers: c, v_0, v_1, a and l (1 ≤ c ≤ 1000, 0 ≤ l < v_0 ≤ v_1 ≤ 1000, 0 ≤ a ≤ 1000) — the length of the book in pages, the initial reading speed, the maximum reading speed, the acceleration in reading speed and the number of pages for rereading.\n\n\n-----Output-----\n\nPrint one integer — the number of days Mister B needed to finish the book.\n\n\n-----Examples-----\nInput\n5 5 10 5 4\n\nOutput\n1\n\nInput\n12 4 12 4 1\n\nOutput\n3\n\nInput\n15 1 100 0 0\n\nOutput\n15\n\n\n\n-----Note-----\n\nIn the first sample test the book contains 5 pages, so Mister B read it right at the first day.\n\nIn the second sample test at first day Mister B read pages number 1 - 4, at second day — 4 - 11, at third day — 11 - 12 and finished the book.\n\nIn third sample test every day Mister B read 1 page of the book, so he finished in 15 days.",
        "Positive": "read = lambda: map(int, input().split())\nc, v0, v1, a, l = read()\ncur = 0\ncnt = 0\nwhile cur < c:\n    cur = max(0, cur - l)\n    cur += min(v1, v0 + a * cnt)\n    cnt += 1\nprint(cnt)"
    },
    {
        "query": "You are given an array $a$ consisting of $n$ integers. In one move, you can jump from the position $i$ to the position $i - a_i$ (if $1 \\le i - a_i$) or to the position $i + a_i$ (if $i + a_i \\le n$).\n\nFor each position $i$ from $1$ to $n$ you want to know the minimum the number of moves required to reach any position $j$ such that $a_j$ has the opposite parity from $a_i$ (i.e. if $a_i$ is odd then $a_j$ has to be even and vice versa).\n\n\n-----Input-----\n\nThe first line of the input contains one integer $n$ ($1 \\le n \\le 2 \\cdot 10^5$) — the number of elements in $a$.\n\nThe second line of the input contains $n$ integers $a_1, a_2, \\dots, a_n$ ($1 \\le a_i \\le n$), where $a_i$ is the $i$-th element of $a$.\n\n\n-----Output-----\n\nPrint $n$ integers $d_1, d_2, \\dots, d_n$, where $d_i$ is the minimum the number of moves required to reach any position $j$ such that $a_j$ has the opposite parity from $a_i$ (i.e. if $a_i$ is odd then $a_j$ has to be even and vice versa) or -1 if it is impossible to reach such a position.\n\n\n-----Example-----\nInput\n10\n4 5 7 6 7 5 4 4 6 4\n\nOutput\n1 1 1 2 -1 1 1 3 1 1",
        "Positive": "from heapq import heappush, heappop\n\nn = int(input())\na = list(map(int, input().split()))\nedge = [[] for _ in range(n)]\nrev = [[] for _ in range(n)]\ninf = 10**9\ncost = [inf]*n\nhq = []\n\nfor i, x in enumerate(a):\n    if i+x < n:\n        edge[i].append(i+x)\n        rev[i+x].append(i)\n        if (a[i] ^ a[i+x]) & 1:\n            cost[i] = 1\n    if i-x >= 0:\n        edge[i].append(i-x)\n        rev[i-x].append(i)\n        if (a[i] ^ a[i-x]) & 1:\n            cost[i] = 1\n\n    if cost[i] == 1:\n        hq.append((1, i))\n\nwhile hq:\n    c, v = heappop(hq)\n    if cost[v] < c:\n        continue\n    c += 1\n    for dest in rev[v]:\n        if cost[dest] > c:\n            cost[dest] = c\n            heappush(hq, (c, dest))\n\nfor i in range(n):\n    if cost[i] == inf:\n        cost[i] = -1\n\nprint(*cost)\n"
    }
]

'''