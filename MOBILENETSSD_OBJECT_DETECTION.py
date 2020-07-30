# Step - 1 : Import required modules
import numpy as np
import cv2

# Step - 2 : Load preprocess and convert image to blob
img_to_detect = cv2.imread('images/testing/scene2.jpg')
img_height = img_to_detect.shape[0]
img_width = img_to_detect.shape[1]
resized_img_to_detect = cv2.resize(img_to_detect, (300, 300))
img_blob = cv2.dnn.blobFromImage(resized_img_to_detect, 0.007843, (300, 300), 127.5)

# Step - 3 : Set class labels
class_labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                "tvmonitor"]

# Step - 4 : Load pre-trained model and get prediction
mobilenetssd = cv2.dnn.readNetFromCaffe('dataset/mobilenetssd.prototext', 'dataset/mobilenetssd.caffemodel')
mobilenetssd.setInput(img_blob)
obj_detection = mobilenetssd.forward()

# Step -5 : Loop over detections, get class label, box coordinates
# loop over detection
no_of_detections = obj_detection.shape[2]

for index in range(0, no_of_detections):
    prediction_confidence = obj_detection[0, 0, index, 2]
    # take predictions with confidence more than 20%
    if prediction_confidence > 0.2:
        # get the prediction label
        predicted_class_index = int(obj_detection[0, 0, index, 1])
        predicted_class_label = class_labels[predicted_class_index]
        # obtain the bounding box co-ordinates for actual image from resized image size
        bounding_box = obj_detection[0, 0, index, 3:7] * np.array([img_width, img_height, img_width, img_height])
        (start_x_pt, start_y_pt, end_x_pt, end_y_pt) = bounding_box.astype('int')

        #Step - 6 : Draw rectangles and text,display the image.
        predicted_class_label = '{}: {:.2f}%'.format(class_labels[predicted_class_index],prediction_confidence*100)
        print('predicted object {}: {}'.format(index+1,predicted_class_label))

        #Draw rectangle and text in the image
        cv2.rectangle(img_to_detect,(start_x_pt,start_y_pt),(end_x_pt,end_y_pt),(0,255,0),2)
        cv2.putText(img_to_detect,predicted_class_label,(start_x_pt,start_y_pt-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

cv2.imshow('Detection Output',img_to_detect)
cv2.waitKey(0)