# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 10:06:07 2018

@author: a
"""
##%===============================Autonomous driving - Car detection
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

%matplotlib inline


#%%--------------------1.Problem Statement
"""
y = (p_c,b_x,b_y,b_h,b_w,c)
"""
#%%------------------2.YOLO
#2.1 model detail
"""
First things to know: 
- The input is a batch of images of shape (m, 608, 608, 3) 
- The output is a list of bounding boxes along with the recognized classes. 
  Each bounding box is represented by 6 numbers (pc,bx,by,bh,bw,c) as explained above. 
  If you expand c into an 80-dimensional vector, each bounding box is then represented by 85 numbers.

We will use 5 anchor boxes. So you can think of the YOLO architecture as the following: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).

If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object.
Since we are using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.
For simplicity, we will flatten the last two last dimensions of the shape (19, 19, 5, 85) encoding. So the output of the Deep CNN is (19, 19, 425).

"""
#2.2 - Filtering with a threshold on class scores
"""
You are going to apply a first filter by thresholding. You would like to get rid of any box for which the class “score” is less than a chosen threshold.

The model gives you a total of 19x19x5x85 numbers, with each box described by 85 numbers. It’ll be convenient to rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables: 
- box_confidence: tensor of shape (19×19,5,1)) containing pcpc (confidence probability that there’s some object) for each of the 5 boxes predicted in each of the 19x19 cells. 
- boxes: tensor of shape (19×19,5,4) containing (bx,by,bh,bw) for each of the 5 boxes per cell. 
- box_class_probs: tensor of shape (19×19,5,80) containing the detection probabilities (c1,c2,...c80) for each of the 80 classes for each of the 5 boxes per cell.
"""
def yolo_filter_boxes(box_confidence,boxes,box_class_probs,threshold = 0.6):
    """
    Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    box_scores = box_confidence * box_class_probs
    
    box_classes = K.argmax(box_scores,axis=-1)
    box_class_scores = K.max(box_scores,axis=-1,keepdims=False)
    
    filtering_mask = box_class_scores >= threshold
    
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)
    
    return scores,boxes,classes
"""    
with tf.Session() as test_a:
    box_confidence = tf.random_normal([19,19,5,1],mean=1,stddev=4,seed=1)
    boxes = tf.random_normal([19,19,5,4],mean=1,stddev=4,seed=1)
    box_class_probs = tf.random_normal([19,19,5,80],mean=1,stddev=4,seed=1)
    scores,boxes,classes = yolo_filter_boxes(box_confidence,boxes,box_class_probs,threshold=0.5)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))
"""

#%%----------------2.3 - Non-max suppression
#intersection over union:IOU
"""
调整编码方式：(x1,y1)对应盒子的左上角，(x2,y2)对应盒子的右下角
xi1,yi1:分别为两个盒子中x1,y1的最大值(意思是更偏向右下边的点)
xi2,yi2:分别为两个盒子中x2,y2的最小值(意思是更偏向左上边的点)
===>>(xi1,yi1),(xi2,yi2):分别对应交集区域的左上角和右下角
"""
def iou(box1,box2):
    """
    boxi:[upleftx1,uplefty1,downrightx2,downrighty2]
    """
    xi1 = max(box1[0],box2[0])
    yi1 = max(box1[1],box2[1])
    xi2 = min(box1[2],box2[2])
    yi2 = min(box1[3],box2[3])
    inter_area = (xi2-xi1)*(yi2-yi1)
    
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    
    return iou
"""    
box1 = (2,1,4,3)
box2 = (1,2,3,4)
print("iou = "+str(iou(box1,box2)))
"""

"""
ou are now ready to implement non-max suppression. The key steps are: 
1. Select the box that has the highest score. 
2. Compute its overlap with all other boxes, and remove boxes that overlap it more than iou_threshold. 
3. Go back to step 1 and iterate until there’s no more boxes with a lower score than the current selected box.
"""
def yolo_non_max_suppression(scores,boxes,classes,max_boxes=10,iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    max_boxes_tensor = K.variable(max_boxes,dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    
    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes,iou_threshold,name = None)
    
    scores = K.gather(scores,nms_indices)
    boxes = K.gather(boxes,nms_indices)
    classes = K.gather(classes,nms_indices)
    
    return scores,boxes,classes
"""
with tf.Session() as test_b:
    scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
    classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))
"""

#%%-------------------------2.4 Wrapping up the filtering
def yolo_eval(yolo_outputs,image_shape=(720.,1280.),max_boxes=10,iou_threshold=0.5,score_threshold=0.6):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    box_confidence,box_xy,box_wh,box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy,box_wh)
    scores,boxes,classes = yolo_filter_boxes(box_confidence,boxes,box_class_probs,score_threshold)
    
    boxes = scale_boxes(boxes,image_shape)
    
    scores,boxes,classes = yolo_non_max_suppression(scores,boxes,classes,max_boxes,iou_threshold)
    
    return scores,boxes,classes
"""    
with tf.Session() as test_b:
    yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    scores, boxes, classes = yolo_eval(yolo_outputs)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))
"""


"""
Summary for YOLO: 
- Input image (608, 608, 3) 
- The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output. 
- After flattening the last two dimensions, the output is a volume of shape (19, 19, 425): 
- Each cell in a 19x19 grid over the input image gives 425 numbers. 
- 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture. 
- 85 = 5 + 80 where 5 is because (pc,bx,by,bh,bw)(pc,bx,by,bh,bw) has 5 numbers, and and 80 is the number of classes we’d like to detect 
- You then select only few boxes based on: 
- Score-thresholding: throw away boxes that have detected a class with a score less than the threshold 
- Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes 
- This gives you YOLO’s final output.
"""


#%%---------------------3 - Test YOLO pretrained model on images
#In this part, you are going to use a pretrained model and test it on the car detection dataset. As usual, you start by creating a session to start your graph. Run the following cell.
sess = K.get_session()

#3.1 - Defining classes, anchors and image shape.
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)   

#3.2 - Loading a pretrained model
yolo_model = load_model("model_data/yolo.h5")

#3.3 - Convert output of the model to usable bounding box tensors
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#3.4 - Filtering boxes
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

#3.5 - Run the graph on an image
image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes

    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    ### START CODE HERE ### (≈ 1 line)
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict = {yolo_model.input:image_data, K.learning_phase(): 0})
    ### END CODE HERE ###

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)

    return out_scores, out_boxes, out_classes

out_scores, out_boxes, out_classes = predict(sess, "test.jpg")













