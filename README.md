# VIPDeepFakeDetection
DeepFake detection from VIP

Continuation of VIP DeepFake Detection Spring 2023.

Progress to Date: Created model for a Faster RCNN -> EfficientNetB0 -> LSTM for deepfake detection
Started outlining new architecture using ViT

Future Work:
Finish ViT architecture. Specifically, Cross ViT where a frame is passed through Faster RCNN, then into an EfficientNet, then into a transformer encoder. This is done for two or three
frames and the tokens from those two or three encoders are passed into a cross-attention layer before classification. 

Design is inspired by Cross Efficient VIT: https://arxiv.org/pdf/2107.02612.pdf

Potential future alternative: More properly incorporate design from the above paper, and have multi-tiered cross attention. Analyze two patch sizes per frame, with results passed through
cross attention, then pass results through cross attention for three frames. 

Even more improvement: Which three frames are the best to look at? Some parts of the video will be consistent frame to frame but others less so. Can a method quickly determine the most
variation in features from efficientnet and then pick the three frames that have the most variation?
