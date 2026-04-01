import java.util.*;
public class Main{
    public static void main(String[] args) {
        try {
            Scanner in=new Scanner(System.in);
            int x=in.nextInt();
            System.out.println(x);
            in.close();
        } catch (InputMismatchException e) {
            System.out.println("helll");
        }  
    }
}