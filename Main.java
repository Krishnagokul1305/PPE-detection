// import java.io.BufferedReader;
// import java.io.InputStreamReader;
import java.util.*;
public class Main{
    public static void main(String[] args) throws Exception{
        try {
           Scanner in=new Scanner(System.in);
           System.out.println(judgeCircle(in.nextLine()));
           
           in.close();
        } catch (InputMismatchException e) {
            System.out.println(e.getMessage());
        }  
    }

     public static boolean judgeCircle(String moves) {
        int x=0,y=0;
        for(int i=0;i<moves.length();i++){
          char cur=moves.charAt(i);
          if(cur=='U') y++;
          if(cur=='D') y--;
          if(cur=='R') x++;
          if(cur=='L') x--;
        }
        return x==0 && y==0;
    }
}