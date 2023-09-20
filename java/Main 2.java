import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Scanner;
import java.util.regex.Pattern;

public class Main2 {

    static final Scanner sc = new Scanner(System.in);

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int N1 = sc.nextInt();
        int N2 = sc.nextInt();
        int M = sc.nextInt();
        HashMap<Integer, ArrayList<Integer>> graph = new HashMap<>();
        for (int i = 0; i < N1 + N2; i++) {
            graph.put(i, new ArrayList<>());
        }
        for (int i = 0; i < M; i++) {
            int a = sc.nextInt() - 1;
            int b = sc.nextInt() - 1;
            graph.get(a).add(b);
            graph.get(b).add(a);
        }
        HashMap<Integer, Integer> history = new HashMap<>();
        history.put(0, 0);
        history.put(N1 + N2 - 1, 0);

        int max = bfs(0, graph, history);
        history = new HashMap<>();
        int ans = max + bfs(N1 + N2 - 1, graph, history);

        System.out.println(ans + 1);

        sc.close();
    }

    private static int bfs(int start, HashMap<Integer, ArrayList<Integer>> graph, HashMap<Integer, Integer> history) {
        Queue<Integer> queue = new LinkedList<>();
        queue.add(start);

        while (!queue.isEmpty()) {
            int current = queue.poll();
            int count = history.get(current) + 1;

            for (int next : graph.get(current)) {
                if (!history.containsKey(next) || history.get(next) > count) {
                    history.put(next, count);
                    queue.add(next);
                }
            }
        }

        int max = Integer.MIN_VALUE;
        for (int value : history.values()) {
            if (value > max) {
                max = value;
            }
        }

        return max;
    }
    /*
     * int K = sc.nextInt();
     * BigInteger bk =new BigInteger(String.valueOf(K));
     * 
     * long[] a = new long[N];
     * BigInteger drink = new BigInteger("0");
     * BigInteger day = new BigInteger("0");
     * 
     * HashMap<Long,Long> map = new HashMap<>();
     * for(int i=0;i<N;i++) {
     * a[i] = sc.nextLong();
     * long b = sc.nextLong();
     * BigInteger added = new BigInteger(String.valueOf(b));
     * drink = drink.add(added);
     * 
     * map.put(a[i], b);
     * }
     * sort(a,0,a.length-1);
     * //println(drink.toString());
     * boolean flag = true;
     * for(int i=0;i<N;i++) {
     * if(drink.compareTo(bk)<0) {
     * if(day.equals(BigInteger.ZERO)) {
     * print(1);
     * }else {
     * print(day.toString());
     * }
     * flag=false;
     * 
     * break;
     * }else {
     * BigInteger subed = new BigInteger(String.valueOf(map.get(a[i])));
     * drink = drink.subtract(subed);
     * BigInteger added = new BigInteger(String.valueOf(a[i]));
     * day = day.add(added);
     * }
     * 
     * }
     * if(flag) {
     * print(day.toString());
     * }
     * 
     * 
     */

    private static void search(int i, HashMap<Integer, ArrayList<Integer>> graph, HashMap<Integer, Integer> history,
            int count) {
        // TODO 自動生成されたメソッド・スタブ
        count++;
        for (int next : graph.get(i)) {
            if (!history.containsKey(next) || history.get(next) > count) {
                history.put(next, count);
                search(next, graph, history, count);
            }
        }

    }

    private static void judge(boolean flag) {
        if (flag) {
            printYes();
        } else {
            printNo();
        }

    }

    private static char nextChar() {
        return sc.next().toCharArray()[0];
    }

    private static int upperBound(int[] list, int value) {
        int first = 0;
        int last = list.length;
        while (first < last) {
            int mid = first + (last - first) / 2;
            if (list[mid] <= value) {
                first = mid + 1;
            } else {
                last = mid;
            }
        }
        return first;
    }

    static class UnionFind {
        private int[] parents;

        // N+1で代入すること
        public UnionFind(int n) {
            parents = new int[n];
            Arrays.fill(parents, -1);
        }

        /**
         * グラフの根を探すメソッド
         * 
         * @param x 根を探したいノード
         * @return 根のノード
         */
        public int find(int x) {
            if (parents[x] < 0) {
                return x;
            } else {
                parents[x] = find(parents[x]);
                return parents[x];
            }
        }

        /**
         * ノードとノードを合体するメソッド
         * 
         * @param x 合体したいノード
         * @param y 合体したいノード
         */
        public void union(int x, int y) {
            x = find(x);
            y = find(y);

            if (x == y) {
                return;
            }

            parents[x] += parents[y];
            parents[y] = x;
        }

        /**
         * xを含むノードのサイズを返すメソッド
         * 
         * @param x サイズを測定したいグラフのノード
         * @return xを含むノードのサイズ
         */
        public int size(int x) {
            return -parents[find(x)];
        }

        /**
         * xとyが同じグラフに属しているか判定するメソッド
         * 
         * @param x
         * @param y
         * @return
         */
        public boolean same(int x, int y) {
            return find(x) == find(y);
        }

        /**
         * xのグラフのメンバーを返すメソッド
         * 
         * @param x
         * @return
         */
        public List<Integer> getMembers(int x) {
            int root = find(x);
            List<Integer> members = new ArrayList<>();
            for (int i = 0; i < parents.length; i++) {
                if (find(i) == root) {
                    members.add(i);
                }
            }
            return members;
        }

        /**
         * ルートの集合を返すメソッド
         * 
         * @return
         */
        public List<Integer> getRoots() {
            List<Integer> roots = new ArrayList<>();
            for (int i = 0; i < parents.length; i++) {
                if (parents[i] < 0) {
                    roots.add(i);
                }
            }
            return roots;
        }

        /**
         * グラフ数を返すメソッド
         * 
         * @return
         */
        public int groupCount() {
            return getRoots().size();
        }

        /**
         * 全てのグラフを返すメソッド
         * 
         * @return
         */
        public Map<Integer, List<Integer>> allGroupMembers() {
            Map<Integer, List<Integer>> groupMembers = new HashMap<>();
            for (int member = 0; member < parents.length; member++) {
                int root = find(member);
                if (!groupMembers.containsKey(root)) {
                    groupMembers.put(root, new ArrayList<>());
                }
                groupMembers.get(root).add(member);
            }
            return groupMembers;
        }
    }

    static class Pair<T, U> {
        T first;
        U second;

        public Pair(T first, U second) {
            this.first = first;
            this.second = second;
        }
    }

    public static int abs(int num) {
        return Math.abs(num);
    }

    public static void yesEnd() {
        printYes();
        System.exit(0);
    }

    public static void noEnd() {
        printNo();
        System.exit(0);
    }

    public static List<List<Long>> combination(long[] a, int K) {
        List<List<Long>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), a, K, 0);
        return result;
    }

    private static void backtrack(List<List<Long>> result, List<Long> tempList, long[] A, int K, int start) {
        if (tempList.size() == K) {
            result.add(new ArrayList<>(tempList));
        } else {
            for (int i = start; i < A.length; i++) {
                tempList.add(A[i]);
                backtrack(result, tempList, A, K, i);
                tempList.remove(tempList.size() - 1);
            }
        }
    }

    public static List<List<Integer>> combination(int[] a, int K) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), a, K, 0);
        return result;
    }

    private static void backtrack(List<List<Integer>> result, List<Integer> tempList, int[] A, int K, int start) {
        if (tempList.size() == K) {
            result.add(new ArrayList<>(tempList));
        } else {
            for (int i = start; i < A.length; i++) {
                tempList.add(A[i]);
                backtrack(result, tempList, A, K, i);
                tempList.remove(tempList.size() - 1);
            }
        }
    }

    private static int[][] inputInt2(int H, int W) {
        // TODO 自動生成されたメソッド・スタブ
        int[][] A = new int[H][W];
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                A[i][j] = sc.nextInt();
            }
        }
        return A;
    }

    private static long[][] inputLong2(int H, int W) {
        // TODO 自動生成されたメソッド・スタブ
        long[][] A = new long[H][W];
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                A[i][j] = sc.nextInt();
            }
        }
        return A;
    }

    private static char[][] inputChar2(int H, int W) {
        // TODO 自動生成されたメソッド・スタブ
        char[][] A = new char[H][W];
        for (int i = 0; i < H; i++) {
            A[i] = inputChar();
        }
        return A;
    }

    private static char[] inputChar() {
        return sc.next().toCharArray();
    }

    private static void print(double d) {
        // TODO 自動生成されたメソッド・スタブ
        System.out.print(d);
    }

    private static void println(double d) {
        // TODO 自動生成されたメソッド・スタブ
        System.out.println(d);
    }

    public static HashMap<Integer, ArrayList<Integer>> makeGraph(int N, int M) {
        HashMap<Integer, ArrayList<Integer>> graph = new HashMap<>();
        for (int i = 1; i <= N; i++) {
            graph.put(i, new ArrayList<>());

        }
        for (int i = 0; i < M; i++) {

            int A = sc.nextInt();
            int B = sc.nextInt();
            graph.get(A).add(B);
            graph.get(B).add(A);
        }
        return graph;
    }

    public static void ln() {
        System.out.println();
    }

    public static void sort2(int[] array, int left, int right, int[] B) {
        if (left <= right) {
            int p = array[(left + right) >>> 1];
            int l = left;
            int r = right;
            while (l <= r) {
                while (array[l] < p) {
                    l++;
                }
                while (array[r] > p) {
                    r--;
                }
                if (l <= r) {
                    int tmp = array[l];
                    array[l] = array[r];
                    array[r] = tmp;
                    int tmp2 = B[l];
                    B[l] = B[r];
                    B[r] = tmp2;
                    l++;
                    r--;
                }
            }
            Main.sort2(array, left, r, B);
            Main.sort2(array, l, right, B);
        }
    }

    public static void print(boolean i) {
        System.out.print(i);
    }

    public static void println(boolean i) {
        System.out.println(i);
    }

    public static void print(char i) {
        System.out.print(i);
    }

    public static void println(char i) {
        System.out.println(i);
    }

    public static void printTime() {
        // 処理前の時刻を取得
        long startTime = System.currentTimeMillis();

        // 時間計測をしたい処理

        // 処理後の時刻を取得
        long endTime = System.currentTimeMillis();

        System.out.println("開始時刻：" + startTime + " ms");
        System.out.println("終了時刻：" + endTime + " ms");
        System.out.println("処理時間：" + (endTime - startTime) + " ms");
    }

    public static List<String> getUpperAlphabets(int K) {
        // final int ALPHABET_SIZE = 'Z' - 'A';
        int ALPHABET_SIZE = K;
        char alphabet = 'A';

        List<String> upperAlphabets = new ArrayList<String>();
        for (int i = 0; i <= ALPHABET_SIZE; i++) {
            upperAlphabets.add(String.valueOf(alphabet++));
        }
        return upperAlphabets;
    }

    public static boolean isAlphabet(char c) {
        String s = String.valueOf(c);
        boolean result = false;
        if (s != null) {
            Pattern pattern = Pattern.compile("^[A-Z]+$");
            result = pattern.matcher(s).matches();
        }
        return result;
    }

    public static void print(int i) {
        System.out.print(i);
    }

    public static void println(int i) {
        System.out.println(i);
    }

    public static void print(long i) {
        System.out.print(i);
    }

    public static void println(long i) {
        System.out.println(i);
    }

    public static void print(String st) {
        System.out.print(st);
    }

    public static void println(String st) {
        System.out.println(st);
    }

    public static void printIndex(int[] ans) {
        for (int o : ans) {
            System.out.print(o + " ");
        }
        System.out.println();
    }

    public static int[] inputInt(int N) {
        int[] a = new int[N];
        for (int i = 0; i < N; i++) {
            a[i] = sc.nextInt();

        }
        return a;
    }

    public static double[] inputDouble(int N) {
        double[] a = new double[N];
        for (int i = 0; i < N; i++) {
            a[i] = sc.nextDouble();

        }
        return a;
    }

    public static String[] inputString(int N) {
        String[] a = new String[N];
        for (int i = 0; i < N; i++) {
            a[i] = sc.next();
            println(a[i]);
        }

        return a;
    }

    public static long[] inputLong(int N) {

        long[] a = new long[N];
        for (int i = 0; i < N; i++) {
            a[i] = sc.nextLong();

        }
        return a;
    }

    public static void sortLong(Long[] array, int left, int right) {
        if (left <= right) {
            Long p = array[(left + right) >>> 1];
            int l = left;
            int r = right;
            while (l <= r) {
                while (array[l].compareTo(p) < 0) {
                    l++;
                }
                while (array[r].compareTo(p) > 0) {
                    r--;
                }
                if (l <= r) {
                    Long tmp = array[l];
                    array[l] = array[r];
                    array[r] = tmp;
                    l++;
                    r--;
                }
            }
            Main.sortLong(array, left, r);
            Main.sortLong(array, l, right);
        }
    }

    public static void sort(long[] array, int left, int right) {
        if (left <= right) {
            long p = array[(left + right) >>> 1];
            int l = left;
            int r = right;
            while (l <= r) {
                while (array[l] < p) {
                    l++;
                }
                while (array[r] > p) {
                    r--;
                }
                if (l <= r) {
                    long tmp = array[l];
                    array[l] = array[r];
                    array[r] = tmp;
                    l++;
                    r--;
                }
            }
            Main.sort(array, left, r);
            Main.sort(array, l, right);
        }
    }

    public static void sort(int[] array, int left, int right) {
        if (left <= right) {
            int p = array[(left + right) >>> 1];
            int l = left;
            int r = right;
            while (l <= r) {
                while (array[l] < p) {
                    l++;
                }
                while (array[r] > p) {
                    r--;
                }
                if (l <= r) {
                    int tmp = array[l];
                    array[l] = array[r];
                    array[r] = tmp;
                    l++;
                    r--;
                }
            }
            Main.sort(array, left, r);
            Main.sort(array, l, right);
        }
    }

    public static void sort(ArrayList<Integer> array, int left, int right) {
        if (left <= right) {
            int p = array.get((left + right) >>> 1);
            int l = left;
            int r = right;
            while (l <= r) {
                while (array.get(l) < p) {
                    l++;
                }
                while (array.get(r) > p) {
                    r--;
                }
                if (l <= r) {
                    int tmp = array.get(l);
                    array.set(l, array.get(r));
                    array.set(r, tmp);
                    l++;
                    r--;
                }
            }
            Main.sort(array, left, r);
            Main.sort(array, l, right);
        }
    }

    public static void sort(double[] array, int left, int right) {
        if (left <= right) {
            double p = array[(left + right) >>> 1];
            int l = left;
            int r = right;
            while (l <= r) {
                while (array[l] < p) {
                    l++;
                }
                while (array[r] > p) {
                    r--;
                }
                if (l <= r) {
                    double tmp = array[l];
                    array[l] = array[r];
                    array[r] = tmp;
                    l++;
                    r--;
                }
            }
            Main.sort(array, left, r);
            Main.sort(array, l, right);
        }
    }

    public static void sort(Long[] array, int left, int right) {
        if (left <= right) {
            Long p = array[(left + right) >>> 1];
            int l = left;
            int r = right;
            while (l <= r) {
                while (array[l] < p) {
                    l++;
                }
                while (array[r] > p) {
                    r--;
                }
                if (l <= r) {
                    Long tmp = array[l];
                    array[l] = array[r];
                    array[r] = tmp;
                    l++;
                    r--;
                }
            }
            Main.sortLong(array, left, r);
            Main.sortLong(array, l, right);
        }
    }

    public static void printYes() {
        System.out.println("Yes");
    }

    public static void printNo() {
        System.out.println("No");
    }

}
