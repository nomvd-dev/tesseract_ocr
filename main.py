import os
from app import app
import pytesseract
import cv2
import numpy as np
import math
from deskew import determine_skew
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, make_response
from werkzeug.utils import secure_filename
from werkzeug.wrappers import response
import pdfkit
from pdf2docx import Converter

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS_PDF = set(['pdf'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def allowed_file_pdf(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_PDF


@app.route('/')
def upload_form():
    filename = 'no_image.jpg'
    return render_template('index.html', filename=filename)
# first argument is image path
# second argument is pre-train east text detector path
@app.route('/convert', methods=['POST'])
def convert_to_word():
    isConverted = False
    PATH = 'static\\results\\'
    file = request.files['pdf']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['RESULT_FOLDER'], filename))
    p = PATH + filename
    print(p)
    docx_file = PATH + filename.rsplit('.', 1)[0].lower() + '.docx'
    print(docx_file)
    cv = Converter(p)
    cv.convert(docx_file, start=0, end=None)
    cv.close()
    isConverted = True
    flash("Converted !!!")
    flash("Saved !!!")
    return redirect("/")

@app.route('/result', methods=['POST'])
def upload_image():
    PATH = 'static\\uploads\\'
    if 'file' not in request.files:
        # flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        # flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        p = PATH + filename
        print(filename.rsplit('.', 1)[0].lower())
        # flash('Image successfully uploaded and displayed below')
        text = "text"
        # p = PATH + filename
        text = text_scanner(p, 'Model\\east_text_detection.pb')
        print(text)
        flash("Rendered !!!")
        return render_template('index.html', text=text, filename=filename)
    else:
        # flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

orig = None

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")

def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < 0.5:
                continue
            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)

def detect_bounding_box(east_model_path, image, origW, origH, rW, rH, W, H):
    ######## Load pre-train model #########
    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east_model_path)

    print("[INFO] Detecting bounding box...")
    ######## Detect bouding box ##########
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    # loop over the bounding boxes
    startX_min = endX_max = int(origW/2)
    startY_min = endY_max = int(origH/2)

    print("[INFO] Calculating bounding box and checking deskew...")
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * 0.05)
        dY = int((endY - startY) * 0.05)
        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))

        # draw the bounding box on the image
        bdb = orig
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

        if startX_min > startX:
            startX_min = startX

        if startY_min > startY:
            startY_min = startY

        if endX_max < endX:
            endX_max = endX

        if endY_max < endY:
            endY_max = endY
    # cv2.imshow('',bdb)
    return (startX_min, endX_max, startY_min, endY_max)

def cheking_deskew(orig_cut):
    img = orig_cut.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(gray)
    if angle == None:
        angle = 0

    print(angle, end='\n\n')

    old_width, old_height = img.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + \
        abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + \
        abs(np.cos(angle_radian) * old_height)

    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    rotated = cv2.warpAffine(img, rot_mat, (int(round(height)), int(
        round(width))), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, gray

def text_scanner(img_path, east_model_path):
    image = cv2.imread(img_path)
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    (newW, newH) = (640, 640)
    # for both the width and height
    rW = origW / float(newW)
    rH = origH / float(newH)
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    (startX_min, endX_max, startY_min, endY_max) = detect_bounding_box(
        east_model_path, image, origW, origH, rW, rH, W, H)
    startY_min = max(0, startY_min-15)
    startX_min = max(0, startX_min-15)
    endX_max = min(origW, endX_max+15)
    endY_max = min(origW, endY_max+15)
    rotated, gray = cheking_deskew(
        orig[startY_min:endY_max, startX_min:endX_max])
    # cv2.imwrite('/content/drive/MyDrive/CS406 project/img/test.jpg',rotated)
    text = pytesseract.image_to_string(rotated)
    return text


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/pdf', methods=['POST'])
def downloadPDF():
    output = request.form.to_dict()
    rendered = output['content']
    # rendered = render_template("index.html")
    pdf = pdfkit.from_string(rendered, False)
    response = make_response(pdf)
    response.headers["Content-Type"] = "application/pd"
    response.headers["Content-Disposition"] = "attachment;filename=result.pdf"
    return response

@app.route('/word', methods=['POST'])
def downloadWord():
    output = request.form.to_dict()
    rendered = output['content']
    pdf = pdfkit.from_string(rendered, False)
    response = make_response(pdf)
    response.headers["Content-Type"] = "application/pd"
    response.headers["Content-Disposition"] = "attachment;filename=result4.pdf"
    docx_file = 'result4.docx'
    cv = Converter('result4.pdf')
    cv.convert(docx_file, start=0, end=None)
    cv.close()
    # return render_template('index.html')
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
