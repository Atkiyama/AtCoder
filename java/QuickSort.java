
public class QuickSort {

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		int[] array = {8,2,7,5,3};
		printData(array);
		sort(array,0,array.length-1);
		printData(array);
	}
	  static void printData(int[] d) {
	        for(int i = 0; i < d.length; i++) System.out.print(d[i] + " ");
	        System.out.println();
	    }
	public static void sort(int[] array,int left,int right) {
		//leftとrightの真ん中の数値をpivotに設定
		if (left>=right) {
            return;
        }
		int pivot = array[(left+right)/2];
		int l = left;
		int r = right;
		int tmp;
		while(l<=r) {
            while(array[l] < pivot) { 
            	//lをarray[l]<pivotの間はインクリメントし続ける
            	//最終的にlのarray[l]はpivot以上の値を指すようになる
            	l++; 
            }
            while(array[r] > pivot) {
            	//lと同様にarray[r]がpivot以下のrを探す
            	r--; 
            }
            if (l<=r) {
            	//実際に入れ替える
                tmp = array[l]; 
                array[l] = array[r]; 
                array[r] = tmp;
                l++; 
                r--;
            }
        }
		 sort(array,left,r);
         sort(array,l,right);
	}

}
