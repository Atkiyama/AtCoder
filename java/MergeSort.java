import java.util.Arrays;

public class MergeSort {

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		int[] array = {8,2,7,5,3};
		printData(array);
		mergesort(array);
		printData(array);
	}
	  private static void mergesort(int[] array) {
		// TODO 自動生成されたメソッド・スタブ
		int mid = array.length/2;
		int[] left = Arrays.copyOfRange(array, 0, mid);
		int[] right = Arrays.copyOfRange(array, mid+1, array.length);
		mergesort(left);
		mergesort(right);
		merge(array,left,right);
		
	}
	  public static void merge(int[] arr, int[] left, int[] right) {
	        int i = 0; // 左側の配列のインデックス
	        int j = 0; // 右側の配列のインデックス
	        int k = 0; // マージした配列のインデックス

	        // 左側と右側の配列を比較し、マージしてソート
	        while (i < left.length && j < right.length) {
	            if (left[i] <= right[j]) {
	                arr[k] = left[i];
	                i++;
	            } else {
	                arr[k] = right[j];
	                j++;
	            }
	            k++;
	        }

	        // 左側と右側の配列の残りの要素を追加
	        while (i < left.length) {
	            arr[k] = left[i];
	            i++;
	            k++;
	        }

	        while (j < right.length) {
	            arr[k] = right[j];
	            j++;
	            k++;
	        }
	    }
	static void printData(int[] d) {
	        for(int i = 0; i < d.length; i++) System.out.print(d[i] + " ");
	        System.out.println();
	    }
	

}
