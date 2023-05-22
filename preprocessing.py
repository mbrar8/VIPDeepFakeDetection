import tensorflow as tf
import tensorflow_hub as hub
import cv2 as cv
import numpy as np
import json
import os

'''
Author: Maheep Brar
File Description: Preprocessing dataset using Faster RCNN
'''

def iou(box1, box2):
    # Compute Intersection over Union fraction given two boxes
    ymin, xmin, ymax, xmax = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    # Total area
    union = (ymax - ymin)*(xmax - xmin) + (ymax2 - ymin2)*(xmax2 - xmin2)
    # Intersection area (the smallest of the far side - the largest of the near side)
    inter = (min(xmax, xmax2) - max(xmin, xmin2)) * (min(ymax, ymax2) - max(ymin, ymin2))
    print("IoU Calculation: ")
    print(float(inter)/(union-inter))
    return float(inter)/(union-inter)




def preprocess(dataset):
    print("Examining " + dataset)
    # Destination of preprocessed videos
    dest = "/data/mbrar/preprocessed_data/" + dataset + "/"
    # Source for preprocessed videos. dataset refers to specifc subfolder (10 GB chunks)
    src = "/data/mbrar/original_data/" + dataset + "/"
    # Load faster RCNN
    faster_rcnn = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")
    # Load metadata - contains file name and labels
    meta_file = open(src + "metadata.json")

    metadata = json.load(meta_file)
    # Iterate over all files in directory
    for file_name in os.listdir(src):
        print("File: " + file_name)
        if ".mp4" not in file_name:
            # File is the metadata, not video
            continue
        fil = os.path.join(src, file_name)
        # Access video
        video = cv.VideoCapture(fil)
        # Get video fps
        video_fps = video.get(cv.CAP_PROP_FPS)
        # Get video frame size
        frame_size = (int(video.get(3)), int(video.get(4)))
       
        new_frame_size = None

        # Boolean to track whether an output video has been created
        output_created = False
        # Keep track of previous bboxes to match up same people
        prev_boxes = {}
        # Index variable to keep track of multiple people in video
        k = 0
        output_vids = {}

        # Opening the video
        while (video.isOpened()):
            opened, frame = video.read()
            # Frame exists
            if opened == True:
                # Convert frame into suitable input
                ref = cv.resize(frame, (640,640))
                tf_fr = tf.convert_to_tensor(ref, dtype=tf.uint8)
                tf_fr = tf.reshape(tf_fr, [1, 640, 640, 3])
                # Run Faster RCNN on frame
                outputs = faster_rcnn(tf_fr)
                det_boxes = tf.unstack(outputs["detection_boxes"][0])
                det_classes = tf.unstack(outputs["detection_classes"][0])
                det_scores = tf.unstack(outputs["detection_scores"][0])
                # Check all found bboxes
                for i in range(len(det_boxes)):
                    det_box = det_boxes[i]
                    det_class = int(det_classes[i])
                    score = float(det_scores[i])
                    # Only examine bboxes classified as people with high confidence
                    if det_class != 1 or score < 0.75:
                        continue
                    
                    # Get crop inside bbox
                    ymin_f, xmin_f, ymax_f, xmax_f = det_box
                    xmin = int(xmin_f*frame_size[0])
                    ymin = int(ymin_f*frame_size[1])
                    xmax = int(xmax_f*frame_size[0])
                    ymax = int(ymax_f*frame_size[1])
                    crop = frame[ymin:ymax, xmin:xmax].copy()
                    
                    if output_created == False:
                        # Creating new video
                        print("\nNew person " + str(k))
                        new_frame_size = (crop.shape[1], crop.shape[0])
                        output_vids[k] = cv.VideoWriter(dest + str(k) + file_name, cv.VideoWriter_fourcc(*'XVID'), video_fps, new_frame_size)
                        # Saving bbox for comparison to later frames
                        prev_boxes[k] = det_box
                        # Write frame to video
                        output_vids[k].write(crop)
                        k += 1

                    else:
                        # Not the first frame - videos already created
                        for key in prev_boxes.keys():
                            prev_box = prev_boxes[key]
                            # Check if current box has large enough overlap with previous ones
                            print("Check iou")
                            if iou(det_box, prev_box) > 0.8:
                                # Box overlaps with previous one by at least 0.8 so matches
                                # Write frame to video
                                print("IoU matches")
                                crop_shaped = cv.resize(crop, new_frame_size)
                                output_vids[key].write(crop_shaped)
                                prev_boxes[key] = det_box # Update prev_box
                                
                
                # Outside of det box for loop
                # At least one pass through a video has happened so output videos were created
                output_created = True

            else:
                break 

        # Cleanup (close videos)
        cv.destroyAllWindows()
        for key in output_vids.keys():
            output_vids[key].release()
        video.release()


preprocess("dfdc_train_part_0")
