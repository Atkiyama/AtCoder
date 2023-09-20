import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Queue;
import java.util.Scanner;

public class Test {
	static final Scanner sc = new Scanner(System.in);

	 public static int findIndex(int[] A, int k) {
	        int left = 0;
	        int right = A.length - 1;

	        while (left <= right) {
	            int mid = left + (right - left) / 2;

	            if (A[mid] < k) {
	                left = mid + 1;
	            } else if (A[mid] > k) {
	                right = mid - 1;
	            } else {
	                // kと一致する要素が見つかった場合、iを返す
	                return mid;
	            }
	        }

	        // kよりも大きい要素がない場合、kより小さい最大の要素の位置を返す
	        return right;
	    }

	    // テスト用例
	    public static void main(String[] args) {
	        int[] A = {1, 3, 5, 7, 9, 11, 13, 15};
	        int k = 8;
	        int index = findIndex(A, k);
	        System.out.println("i = " + index);
	    }
	private static void printMinus() {
		System.out.print(-1);
	}

	private static int judge(int bit, int N, int W, int[] A) {
		// TODO 自動生成されたメソッド・スタブ
		//足し算の合計
		int S = 0;
		/**
		 *配列Aを全部みる
		 */
		for (int i = 0; i < N; i++) {
			//引数bit(二進数)のN桁目を見てそれが1なら加える
			if ((bit & (1 << i)) == 1) {
				S += A[i];
			}
		}
		if (S == W) {
			return 1;
		} else {
			return 0;
		}
	}

	/**
	 * 幅優先探索でnowから最も離れたノードまでの経路数を返す
	 * @param graph
	 * @param now
	 * @return
	 */
	private static int bfs(HashMap<Integer, ArrayList<Integer>> graph, int now) {
		// TODO 自動生成されたメソッド・スタブ
		HashMap<Integer, Integer> history = new HashMap<>();
		Queue<Integer> q = new LinkedList<>();

		q.add(now);
		history.put(now, 0);
		while (q.size() > 0) {
			int x = q.poll();
			for (int i = 0; i < graph.get(x).size(); i++) {
				int y = graph.get(x).get(i);
				if (!history.containsKey(y)) {
					history.put(y, history.get(x) + 1);
					q.add(y);
				}

			}

		}
		Collection<Integer> values = history.values();
		return Collections.max(values);
	}

	public static class MinHeap<T extends Comparable<T>> {
		private T[] heap;
		private int size;
		private int capacity;

		public MinHeap(int capacity) {
			this.capacity = capacity;
			this.size = 0;
			this.heap = (T[]) new Comparable[capacity];
		}

		public int getSize() {
			return size;
		}

		public boolean isEmpty() {
			return size == 0;
		}

		public void insert(T value) {
			if (size == capacity) {
				throw new IllegalStateException("Heap is full");
			}

			// 新しい要素を末尾に追加
			heap[size] = value;
			size++;

			// ヒープを再構築
			heapifyUp(size - 1);
		}

		/**
		 * 最小値を取り出す
		 * @return
		 */
		public T extractMin() {
			if (isEmpty()) {
				throw new IllegalStateException("Heap is empty");
			}

			T minValue = heap[0];
			heap[0] = heap[size - 1];
			size--;

			// ヒープを再構築
			heapifyDown(0);

			return minValue;
		}

		private void heapifyUp(int index) {
			int parentIndex = (index - 1) / 2;

			if (index > 0 && heap[index].compareTo(heap[parentIndex]) < 0) {
				swap(index, parentIndex);
				heapifyUp(parentIndex);
			}
		}

		private void heapifyDown(int index) {
			int leftChildIndex = 2 * index + 1;
			int rightChildIndex = 2 * index + 2;
			int smallestIndex = index;

			if (leftChildIndex < size && heap[leftChildIndex].compareTo(heap[smallestIndex]) < 0) {
				smallestIndex = leftChildIndex;
			}

			if (rightChildIndex < size && heap[rightChildIndex].compareTo(heap[smallestIndex]) < 0) {
				smallestIndex = rightChildIndex;
			}

			if (smallestIndex != index) {
				swap(index, smallestIndex);
				heapifyDown(smallestIndex);
			}
		}

		private void swap(int i, int j) {
			T temp = heap[i];
			heap[i] = heap[j];
			heap[j] = temp;
		}

		@Override
		public String toString() {
			return Arrays.toString(Arrays.copyOf(heap, size));
		}
	}

	public static class Tuple<A extends Comparable<A>, B extends Comparable<B>> implements Comparable<Tuple<A, B>> {
		private A first;
		private B second;

		public Tuple(A first, B second) {
			this.first = first;
			this.second = second;
		}

		public A getFirst() {
			return first;
		}

		public B getSecond() {
			return second;
		}

		@Override
		public boolean equals(Object o) {
			if (this == o)
				return true;
			if (o == null || getClass() != o.getClass())
				return false;
			Tuple<?, ?> tuple = (Tuple<?, ?>) o;
			return Objects.equals(first, tuple.first) && Objects.equals(second, tuple.second);
		}

		@Override
		public int hashCode() {
			return Objects.hash(first, second);
		}

		@Override
		public int compareTo(Tuple<A, B> other) {
			if (first.compareTo(other.getFirst()) != 0) {
				return first.compareTo(other.getFirst());
			}
			return second.toString().compareTo(other.getSecond().toString());
		}

		@Override
		public String toString() {
			return "(" + first + ", " + second + ")";
		}
	}

	public static <T extends Comparable<T>> void sort(List<T> list, boolean isQuick) {
		if (isQuick) {
			quickSort(list);
		} else {
			mergeSort(list, 0, list.size() - 1);
		}
	}

	public static <T extends Comparable<T>> void quickSort(List<T> list) {
		quickSortHelper(list, 0, list.size() - 1);
	}

	private static <T extends Comparable<T>> void quickSortHelper(List<T> list, int left, int right) {
		if (left < right) {
			int pivotIndex = partition(list, left, right);
			quickSortHelper(list, left, pivotIndex - 1);
			quickSortHelper(list, pivotIndex + 1, right);
		}
	}

	private static <T extends Comparable<T>> int partition(List<T> list, int left, int right) {
		T pivot = list.get(right);
		int i = left - 1;

		for (int j = left; j < right; j++) {
			if (list.get(j).compareTo(pivot) <= 0) {
				i++;
				swap(list, i, j);
			}
		}

		swap(list, i + 1, right);
		return i + 1;
	}

	private static <T> void swap(List<T> list, int i, int j) {
		T temp = list.get(i);
		list.set(i, list.get(j));
		list.set(j, temp);
	}

	public static <T extends Comparable<T>> void mergeSort(List<T> list, int left, int right) {
		if (left < right) {
			int mid = (left + right) / 2;
			mergeSort(list, left, mid);
			mergeSort(list, mid + 1, right);
			merge(list, left, mid, right);
		}
	}

	public static <T extends Comparable<T>> void merge(List<T> list, int left, int mid, int right) {
		List<T> leftList = new ArrayList<>();
		List<T> rightList = new ArrayList<>();

		for (int i = left; i <= mid; i++) {
			leftList.add(list.get(i));
		}

		for (int j = mid + 1; j <= right; j++) {
			rightList.add(list.get(j));
		}

		int i = 0, j = 0;
		int k = left;

		while (i < leftList.size() && j < rightList.size()) {
			if (leftList.get(i).compareTo(rightList.get(j)) <= 0) {
				list.set(k, leftList.get(i));
				i++;
			} else {
				list.set(k, rightList.get(j));
				j++;
			}
			k++;
		}

		while (i < leftList.size()) {
			list.set(k, leftList.get(i));
			i++;
			k++;
		}

		while (j < rightList.size()) {
			list.set(k, rightList.get(j));
			j++;
			k++;
		}
	}

	public static void sort(int[] array, boolean isQuick) {
		if (isQuick) {
			quickSort(array, 0, array.length - 1);
		} else {
			mergeSort(array, 0, array.length - 1);
		}
	}

	public static void sort(double[] array, boolean isQuick) {
		if (isQuick) {
			quickSort(array, 0, array.length - 1);
		} else {
			mergeSort(array, 0, array.length - 1);
		}
	}

	public static void sort(long[] array, boolean isQuick) {
		if (isQuick) {
			quickSort(array, 0, array.length - 1);
		} else {
			mergeSort(array, 0, array.length - 1);
		}
	}

	public static void quickSort(int[] arr, int low, int high) {
		if (low < high) {
			int pivotIndex = partition(arr, low, high);
			quickSort(arr, low, pivotIndex - 1);
			quickSort(arr, pivotIndex + 1, high);
		}
	}

	public static void quickSort(double[] arr, int low, int high) {
		if (low < high) {
			int pivotIndex = partition(arr, low, high);
			quickSort(arr, low, pivotIndex - 1);
			quickSort(arr, pivotIndex + 1, high);
		}
	}

	public static void quickSort(long[] arr, int low, int high) {
		if (low < high) {
			int pivotIndex = partition(arr, low, high);
			quickSort(arr, low, pivotIndex - 1);
			quickSort(arr, pivotIndex + 1, high);
		}
	}

	public static int partition(int[] arr, int low, int high) {
		int pivot = arr[high];
		int i = low - 1;

		for (int j = low; j < high; j++) {
			if (arr[j] <= pivot) {
				i++;
				swap(arr, i, j);
			}
		}

		swap(arr, i + 1, high);
		return i + 1;
	}

	public static int partition(double[] arr, int low, int high) {
		double pivot = arr[high];
		int i = low - 1;

		for (int j = low; j < high; j++) {
			if (arr[j] <= pivot) {
				i++;
				swap(arr, i, j);
			}
		}

		swap(arr, i + 1, high);
		return i + 1;
	}

	public static int partition(long[] arr, int low, int high) {
		long pivot = arr[high];
		int i = low - 1;

		for (int j = low; j < high; j++) {
			if (arr[j] <= pivot) {
				i++;
				swap(arr, i, j);
			}
		}

		swap(arr, i + 1, high);
		return i + 1;
	}

	public static void swap(int[] arr, int i, int j) {
		int temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	}

	public static void swap(double[] arr, int i, int j) {
		double temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	}

	public static void swap(long[] arr, int i, int j) {
		long temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	}

	public static void printArray(int[] arr) {
		for (int num : arr) {
			System.out.print(num + " ");
		}
		System.out.println();
	}

	public static void printArray(double[] arr) {
		for (double num : arr) {
			System.out.print(num + " ");
		}
		System.out.println();
	}

	public static void printArray(long[] arr) {
		for (long num : arr) {
			System.out.print(num + " ");
		}
		System.out.println();
	}

	public static void mergeSort(int[] arr, int left, int right) {
		if (left < right) {
			int mid = (left + right) / 2;
			mergeSort(arr, left, mid);
			mergeSort(arr, mid + 1, right);
			merge(arr, left, mid, right);
		}
	}

	public static void mergeSort(double[] arr, int left, int right) {
		if (left < right) {
			int mid = (left + right) / 2;
			mergeSort(arr, left, mid);
			mergeSort(arr, mid + 1, right);
			merge(arr, left, mid, right);
		}
	}

	public static void mergeSort(long[] arr, int left, int right) {
		if (left < right) {
			int mid = (left + right) / 2;
			mergeSort(arr, left, mid);
			mergeSort(arr, mid + 1, right);
			merge(arr, left, mid, right);
		}
	}

	public static void merge(int[] arr, int left, int mid, int right) {
		int n1 = mid - left + 1;
		int n2 = right - mid;

		int[] leftArray = new int[n1];
		int[] rightArray = new int[n2];

		for (int i = 0; i < n1; ++i)
			leftArray[i] = arr[left + i];
		for (int j = 0; j < n2; ++j)
			rightArray[j] = arr[mid + 1 + j];

		int i = 0, j = 0;
		int k = left;

		while (i < n1 && j < n2) {
			if (leftArray[i] <= rightArray[j]) {
				arr[k] = leftArray[i];
				i++;
			} else {
				arr[k] = rightArray[j];
				j++;
			}
			k++;
		}

		while (i < n1) {
			arr[k] = leftArray[i];
			i++;
			k++;
		}

		while (j < n2) {
			arr[k] = rightArray[j];
			j++;
			k++;
		}
	}

	public static void merge(double[] arr, int left, int mid, int right) {
		int n1 = mid - left + 1;
		int n2 = right - mid;

		double[] leftArray = new double[n1];
		double[] rightArray = new double[n2];

		for (int i = 0; i < n1; ++i)
			leftArray[i] = arr[left + i];
		for (int j = 0; j < n2; ++j)
			rightArray[j] = arr[mid + 1 + j];

		int i = 0, j = 0;
		int k = left;

		while (i < n1 && j < n2) {
			if (leftArray[i] <= rightArray[j]) {
				arr[k] = leftArray[i];
				i++;
			} else {
				arr[k] = rightArray[j];
				j++;
			}
			k++;
		}

		while (i < n1) {
			arr[k] = leftArray[i];
			i++;
			k++;
		}

		while (j < n2) {
			arr[k] = rightArray[j];
			j++;
			k++;
		}
	}

	public static void merge(long[] arr, int left, int mid, int right) {
		int n1 = mid - left + 1;
		int n2 = right - mid;

		long[] leftArray = new long[n1];
		long[] rightArray = new long[n2];

		for (int i = 0; i < n1; ++i)
			leftArray[i] = arr[left + i];
		for (int j = 0; j < n2; ++j)
			rightArray[j] = arr[mid + 1 + j];

		int i = 0, j = 0;
		int k = left;

		while (i < n1 && j < n2) {
			if (leftArray[i] <= rightArray[j]) {
				arr[k] = leftArray[i];
				i++;
			} else {
				arr[k] = rightArray[j];
				j++;
			}
			k++;
		}

		while (i < n1) {
			arr[k] = leftArray[i];
			i++;
			k++;
		}

		while (j < n2) {
			arr[k] = rightArray[j];
			j++;
			k++;
		}
	}

	public static int inputInt() {
		return sc.nextInt();
	}

	public static double inputDouble() {
		return sc.nextDouble();
	}

	public static long inputLong() {
		return sc.nextLong();
	}

	private static String[] inputString(int N) {
		String[] s = new String[N];
		for (int i = 0; i < N; i++) {
			s[i] = sc.next();
		}
		return s;
	}

	private static char[] inputChar() {
		return sc.next().toCharArray();
	}

	public static void yesEnd() {
		printYes();
		System.exit(0);
	}

	public static void noEnd() {
		printNo();
		System.exit(0);
	}

	public static HashMap<Integer, ArrayList<Integer>> inputGraph(int N, int M) {
		HashMap<Integer, ArrayList<Integer>> graph = new HashMap<>();
		for (int i = 0; i < N; i++) {
			graph.put(i + 1, new ArrayList<>());
		}
		for (int i = 0; i < M; i++) {
			int a = sc.nextInt();
			int b = sc.nextInt();
			graph.get(a).add(b);
			graph.get(b).add(a);
		}
		return graph;
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

	public static long[] inputLong(int N) {
		long[] a = new long[N];
		for (int i = 0; i < N; i++) {
			a[i] = sc.nextLong();
		}
		return a;
	}

	public static void print(int i) {
		System.out.print(i);
	}

	public static void println(int i) {
		System.out.println(i);
	}

	public static void print(double i) {
		System.out.print(i);
	}

	public static void println(double i) {
		System.out.println(i);
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

	public static void print(long i) {
		System.out.print(i);
	}

	public static void println(long i) {
		System.out.println(i);
	}

	public static void print(String i) {
		System.out.print(i);
	}

	public static void println(String i) {
		System.out.println(i);
	}

	public static void ln() {
		System.out.println();
	}

	public static void printYes() {
		System.out.println("Yes");
	}

	public static void printNo() {
		System.out.println("No");
	}

	public static void judge(boolean flag) {
		if (flag) {
			printYes();
		} else {
			printNo();
		}
	}

}
