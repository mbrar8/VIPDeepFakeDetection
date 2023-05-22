import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as c
from efficient_net import EfficientNetB0


class LSTMDense(nn.Module):

    def __init__(self):
        super().__init__()


        self.l1 = nn.LSTM(15680, 128)
        self.d1 = nn.Linear(128, 32)
        self.d2 = nn.Linear(32, 2)


    def forward(self, x):
        print("LSTM Input shape")
        print(x.shape)
        r, (h,c) = self.l1(x)
        r = nn.ReLU()(self.d1(r[-1]))
        r = self.d2(r)
        return r

        
                

def runModel(vids, labels):
    model = EfficientNetB0()
    model.eval()
    print("Model in eval mode")
    print(len(vids))
    #output_maps = np.empty(shape=(3,3,7,7,320))
    output_maps = []
    for vid in vids:
        print(vid.shape)
        feature_map = []
        #feature_map = np.empty(shape=(3,7,7,320))
        for frame in vid:
            print("Frame type")
            print(frame.dtype)
            print(frame.shape)
            feature_map.append(model(frame.float().permute(2,0,1)).detach().numpy().flatten())
        print("Feature map for one video created")
        print(np.array(feature_map).shape)
        np.save('featuremap', np.array(feature_map))
        output_maps.append(np.array(feature_map))

    lstm = LSTMDense()
    lstm.eval()
    print("LSTM in eval mode")
    for i in range(len(output_maps)):
        print(output_maps[i].dtype)
        print(output_maps[i].shape)
        outputs = lstm(torch.from_numpy(output_maps[i]))
        print("Labeled: " + str(labels[i]))
        print("Model raw logits: " + str(outputs))

    return


#STEPS (Erin Joy and Maheep Brar)

# extract frames 4 5 and 6 from each video we have preprocessed and save as numpy arrays
    # preproccess for just the heads after extracting specified frame
    # note already cropped for the bounding box
# for each video, run efficient net on each of the three frames and save the feature maps
# pass feature maps through the LSTM
# add dense layers after the LSTM for categorization (number of layers depends on layers in LSTM)


#FUNCTIONS (Erin Joy)

# frame extractor --> indexing for frames -- Erin Joy
    # preprocessing (cutting the heads off)
# load frames into numpy array -- Erin Joy
    # one array: array of arrays.  Row=video column=frame (3 ea)
#call efficient net model --> return type? Should be a feature map represented by pytorch matrix -- Maheep
    # pass in one frame at a time
# pass in feature map for a video into LSTM --> return bool after dense layers


# make a class for LSTM dense layers



"""
@Author: Erin Joy Kramer
@Date Started: 4/15/2023
@Date Finished: 

Extract second set of three frames from the video and crop them so that we are only getting the faces of the people in the video.

Current Issue: opened is never true so function hangs
"""
def getFrames(files):
    videos = []
    for fil in files:
        print(fil)
        video = c.VideoCapture(fil)
    
        height, width = video.get(3),video.get(4) 
    
        frame_count = 0 #count number of valid frames found
        frames = []
        #frames=np.empty(shape=(3,width,width)) #hold specified frames that we are looking for
    
        while (frame_count < 6) : #iterate for second set of 3 frames
            
            opened, frame = video.read()
        
            if (opened == True):
                if (frame_count>2): #checking if is second set of three frames
                    frame = c.cvtColor(frame, c.COLOR_BGR2RGB)
                    crop_frame=frame[0:int(width),0:int(width)] #doesn't currently work
                    resized_frame = c.resize(crop_frame, dsize=(224,224), interpolation=c.INTER_LINEAR)
                    frames.append(resized_frame)
                
                frame_count+=1
            else:
                break
            # frame_count+=1  #add so that we don't hang at compilation but lines above should be uncommented and this should not be here
            
        video.release()
        videos.append(torch.from_numpy(np.array(frames)))
    return videos


labels = [1,0,1]
vids = getFrames(['/data/mbrar/preprocessed_data/dfdc_train_part_0/0yotyalfqqv.mp4', '/data/mbrar/preprocessed_data/dfdc_train_part_0/0yokwqbiqle.mp4', '/data/mbrar/preprocessed_data/dfdc_train_part_0/0yminuwlmoh.mp4'])


runModel(vids, labels)
