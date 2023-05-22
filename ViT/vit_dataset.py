import torch
import torch.nn as nn
import os
import json
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

"""
Class that inherits from torch.utils.data.Dataset as a CustomDataset
"""


class DFDCImageDataset(Dataset):
    
    def __init__(self, data_folders, k_frames, transforms):
        # Only save every k frames
        self.k_frames = k_frames
        # Filename list to reference specific video in order
        self.filenames = []
        # Saved labels list
        self.labels = []
        for folder in data_folders:
            # Get the json file with labels for videos
            js = json.load("/data/mbrar/preprocssed_data/" + folder + "/metadata.json")
            for filename in os.listdir("/data/mbrar/preprocessed_data/" + folder):
                f = os.path.join("/data/mbrar/preprocessed_data/" + folder + "/", filename)
                # Non-files or metadata.json is to be ignored
                if ".mp4" not in f:
                    continue
                vid = cv.VideoCapture(f)
                # Ignore any video with less than 10 frames (RCNN sometimes detects non-people)
                if (int(vid.get(cv.CAP_PROP_FRAME_COUNT)) < 10):
                    vid.release()
                    continue

                self.filenames.append(f)
                # Save label as not-fake. I went with 0 for real because this is searching for fakes
                # So 1 or positive makes sense for it is fake
                label = 0
                # If it is labeled FAKE it is 1 (else it is as initialized)
                if (js[filename[1:][label] == "FAKE"):
                    label = 1

                self.labels.append(label)

    def __len__(self):
        # Required method to get length of dataset
        return len(self.labels)


    def __getitem__(self, idx):
        # Method to get a specific item (i.e. frames from certain video) at index idx
        # Get specific file and label
        filename = self.filenames[idx]
        label = self.labels[idx]
        # Open video
        vid = cv.VideoCapture(filename)
        # Get width and height and set frame_ctr for the only use every k frames
        width, height = int(vid.get(3)), int(vid.get(4))
        frame_ctr = 0
        # Array to store frames
        frames = np.array()

        while (vid.isOpened()):
            opened, frame = vid.read()
            if opened == True:
                # If k = 1, frame_ctr is set to 1 immediately (meaning look at every frame)
                frame_ctr += 1
                # Hit the k'th frame
                if frame_ctr == self.k_frames:
                    # Crop the top part of the image with dimensions (wxw) given image (wxh)
                    cropped = frame[0:width,0:width]
                    np.append(frames, cropped)
                    # Reset frame_ctr (otherwise we'd only get 1 frame)
                    frame_ctr = 0

        # Release the video
        vid.release()
        return frames

    
