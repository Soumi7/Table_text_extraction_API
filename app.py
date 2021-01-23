from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
app = Flask(__name__)
app.config['imgdir'] = "imgdir"
import numpy as np
import cv2
import pandas as pd
import pytesseract

# Disable scientific notation for clarity
# Load the model


@app.route('/')
def home():
    return 'Table Data API'


@app.route('/predict_api', methods=["GET","POST"])
def list_post():    
    #data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32) #number of images 1 (RGB image)
    data = request.files['data']
    
    filename = secure_filename(data.filename) # save file 
    filepath = os.path.join(app.config['imgdir'], filename);
    data.save(filepath)
    img = cv2.imread(filepath,0)

    thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
    img_bin = 255-img_bin

    img_bin1 = 255-img
    thresh1,img_bin1_otsu = cv2.threshold(img_bin1,128,255,cv2.THRESH_OTSU)

    img_bin2 = 255-img
    thresh1,img_bin_otsu = cv2.threshold(img_bin2,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[1]//100))
    eroded_image = cv2.erode(img_bin_otsu, vertical_kernel, iterations=3)
    vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=3)

    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1]//100, 1))
    horizontal_lines = cv2.erode(img_bin, hor_kernel, iterations=5)
    horizontal_lines = cv2.dilate(horizontal_lines, hor_kernel, iterations=5)

    vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)
    thresh, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bitxor = cv2.bitwise_xor(img,vertical_horizontal_lines)
    bitnot = cv2.bitwise_not(bitxor)


    contours, hierarchy = cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda x:x[1][1]))

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w<1000 and h<500):
            image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            boxes.append([x,y,w,h])

    rows=[]
    columns=[]
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)
    print(mean)
    columns.append(boxes[0])
    previous=boxes[0]
    for i in range(1,len(boxes)):
        if(boxes[i][1]<=previous[1]+mean/2):
            columns.append(boxes[i])
            previous=boxes[i]
            if(i==len(boxes)-1):
                rows.append(columns)
        else:
            rows.append(columns)
            columns=[]
            previous = boxes[i]
            columns.append(boxes[i])
    print("Rows")
    for row in rows:
        print(row)

    total_cells=0
    for i in range(len(row)):
        if len(row[i]) > total_cells:
            total_cells = len(row[i])
    print(total_cells)

    center = [int(rows[i][j][0]+rows[i][j][2]/2) for j in range(len(rows[i])) if rows[0]]
    print(center)

    center=np.array(center)
    center.sort()
    print(center)

    
    boxes_list = []
    for i in range(len(rows)):
        l=[]
        for k in range(total_cells):
            l.append([])
        for j in range(len(rows[i])):
            diff = abs(center-(rows[i][j][0]+rows[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            l[indexing].append(rows[i][j])
        boxes_list.append(l)
    for box in boxes_list:
        print(box)

    dataframe_final=[]
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            s=''
            if(len(boxes_list[i][j])==0):
                dataframe_final.append(' ')
            else:
                for k in range(len(boxes_list[i][j])):
                    y,x,w,h = boxes_list[i][j][k][0],boxes_list[i][j][k][1], boxes_list[i][j][k][2],boxes_list[i][j][k][3]
                    roi = bitnot[x:x+h, y:y+w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(roi,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel,iterations=1)
                    erosion = cv2.erode(dilation, kernel,iterations=2)                
                    out = pytesseract.image_to_string(erosion)
                    if(len(out)==0):
                        out = pytesseract.image_to_string(erosion)
                    s = s +" "+ out
                dataframe_final.append(s)
    print(dataframe_final)
    arr = np.array(dataframe_final)
    print(arr)    
    dataframe = pd.DataFrame(arr.reshape(len(rows), total_cells))
    data = dataframe.style.set_properties(align="left")
    dataframe.to_csv("output.csv")
    dataframe=pd.read_csv("output.csv")
    return jsonify(result= dataframe.to_json())

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)