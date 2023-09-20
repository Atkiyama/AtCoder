public class Mynavi {
	public static void main(String[] args) {
		//鍵
        final int Ukey = -3;
        final int Lkey = -4;

        //暗号対象の文字列
        String source = "ZlexaDwxlirePisjsyvgRqtDrcalHrmWaDwjsXQhih";

        //アルファベット以外は消去
        byte[] tmp = source.getBytes();
        byte[] buf = new byte[tmp.length];

        for (int i = 0; i < tmp.length; i++) {
            // シフトした文字がA(65)からZ(90)の間に収まるようにする
        	int key;
        	if(isLower(String.valueOf(source.charAt(i)))) {
        		key=Lkey;
        		int num = ((tmp[i] - 97) + key + 26) % 26;
                buf[i] = (byte) (num + 97);
        	}else {
        		key=Ukey;
        		int num = ((tmp[i] - 65) + key + 26) % 26;
                buf[i] = (byte) (num + 65);
        	}
        		
            
        }

        System.out.println((new String(buf)).toLowerCase());
		
	}

	private static boolean isLower(String str) {
		// TODO 自動生成されたメソッド・スタブ
		return str.equals(str.toLowerCase());

	}
	
	

}
