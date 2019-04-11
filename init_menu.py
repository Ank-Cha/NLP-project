import os
import subprocess
import urllib


img = input("Enter image path: ")
print("Choose one of the following:\n 1. quadrilateral text detection\n 2.Text recognition\n 3.Object detection\n 4. "
      "Buy the product (Product and brand detection + Link generation)")

b=0

# looping until a vaild option is selected
while(b==0):
    # Getting option number from user
    a = int(input())
    if(a==1):
        subprocess.call(["python", "quadtri_text_detect.py", "--input", img])
        b=1

    elif(a==2):
        subprocess.call(["python" ,"text_recognition.py" , "--image", img, "--padding", "0.05"])
        b=1
    elif(a==3):
        subprocess.call(["python", "object_detect_yolo.py", "--image", img, "--yolo", "yolo-coco"])
    elif(a==4):
        q = subprocess.call(["python" ,"text_recognition.py", "--image", img, "--padding", "0.05"])
        w = subprocess.call(["python", "object_detect_yolo.py", "--image", img, "--yolo", "yolo-coco"])
        f= open("data.txt","r")
        #the foolowing code gathers the data and generates a query which is added to URL to get the product page
        ip = f.read()
        ip=ip.split(" ")
        sq="+".join(ip)

        url = "https://www.amazon.in/s?k="+sq
        url2="https://dir.indiamart.com/search.mp?ss="+sq
        os.startfile(url)
        os.startfile(url2)
        b=1
        f = open("data.txt","w")
        f.write(" ")
        f.close()
    else:
        print("PLEASE ENTER A VAILD OPTION:")


print("STATUS OK")