from flask import Flask, request, jsonify, render_template, url_for, make_response
from werkzeug.utils import secure_filename
import os
import json
import numpy as np
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./imgdir"
import numpy as np
import cv2
import pandas as pd
import pytesseract
import io
from PIL import Image
#import StringIO
#import csv

# Disable scientific notation for clarity
# Load the model
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_api', methods=["GET","POST"])
def list_post():    
    # #data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32) #number of images 1 (RGB image)
    # data = request.files['data']
    # in_memory_file = io.BytesIO()
    # filepath = os.path.join(app.config['imgdir'], "img")
    # filepath.save(in_memory_file)
    
    # # filename = secure_filename(data.filename) # save file 
    
    # # data.save(filepath)
    # img = cv2.imread(filepath,0)

    #npimg = np.fromfile(request.files['file'], numpy.uint8)
    # convert numpy array to image
    #img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

    file = request.files['file']
    npimg = np.fromfile(file, np.uint8)
    img = cv2.imdecode(npimg, 0)

    thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255-img_bin
    #####################3
    
    kernel_len = np.array(img).shape[1]//100
    
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1)) #cv2 making rectangle
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) #

    
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3) 
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3) 
    #cv2.imwrite("/content/drive/MyDrive/Colab Notebooks/vertical.jpg",vertical_lines)
    
    #Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3) #erosion 
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)  #dilation
    #cv2.imwrite("/content/drive/MyDrive/Colab Notebooks/horizontal.jpg",horizontal_lines)
    

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0) #image : vertical + horizontal : equally add
    #Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imwrite("/content/drive/MyDrive/Colab Notebooks/img_vh.jpg", img_vh)
    bitxor = cv2.bitwise_xor(img,img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    
    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    #Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))] 

    #Get mean of heights
    mean = np.mean(heights)

    #Create list box to store all boxes in  
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w<2000 and h<2500):
            image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            box.append([x,y,w,h])
            
    #Creating two lists to define row and column in which cell is located
    row=[]
    column=[]
    j=0
    print(box)
    #Sorting the boxes to their respective row and column
    for i in range(len(box)):    
            
        if i==0:
            column.append(box[i])
            previous=box[i]    
        
        else:
            if(box[i][1]<=previous[1]+mean/2):
                column.append(box[i])
                previous=box[i]            
                
                if(i==len(box)-1):
                    row.append(column)        
                
            else:
                row.append(column)
                column=[]
                previous = box[i]
                column.append(box[i])
                
    print(column)
    print(row)

    #calculating maximum number of cells
    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol

    #Retrieving the center of each column
    center=[int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]

    center=np.array(center)
    center.sort()
    print(center)
    #Regarding the distance to the columns center, the boxes are arranged in respective order

    finalboxes = []
    for i in range(len(row)):
        lis=[]
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)


    #from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    outer=[]
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner=''
            if(len(finalboxes[i][j])==0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                    finalimg = bitnot[x:x+h, y:y+w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalimg,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel,iterations=1)
                    erosion = cv2.erode(dilation, kernel,iterations=2)                
                    out = pytesseract.image_to_string(erosion)
                    if(len(out)==0):
                        out = pytesseract.image_to_string(erosion, config='--psm 3')
                    inner = inner +" "+ out
                outer.append(inner)

        


    ##############################3
    arr = np.array(outer)
    # print(arr)    
    dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
    data = dataframe.style.set_properties(align="left")
    # dataframe.to_csv("output.csv")
    # dataframe=pd.read_csv("output.csv")

    resp = make_response(dataframe.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=table.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)