
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import math
from Pipeline import Pipeline


class ReadFrame(Pipeline):

    def __init__(self,filename):
        self.video_in = cv2.VideoCapture(filename)


    def generator(self):
        while(True):
            ret, frame = self.video_in.read()
            if ret == True:
                yield frame
            else:
                raise StopIteration 

class ConvertToGray(Pipeline):
    def map(self,frame):
        gray_image = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        return gray_image

class EdgeDetect(Pipeline):
    def map(self,frame):
        cannyed_image = cv2.Canny(frame,100,200)
        return cannyed_image

class CropFrame(Pipeline):
    def region_of_interest(self, frame, vertices):
        mask = np.zeros_like(frame)
        match_mask_color = 255
        cv2.fillPoly(mask,vertices,match_mask_color)
        masked_image = cv2.bitwise_and(frame,mask)
        return masked_image

    def map(self,frame):     

        height = frame.shape[0]
        width = frame.shape[1]

        region_of_interest_vertices = [
            (0,height),
            (width/2,height/2),
            (width,height)
        ]      

        cropped_image = self.region_of_interest(
            frame,
            np.array([region_of_interest_vertices], np.int32)
        )
        return cropped_image


class DetectLine(Pipeline):
    def map(self,frame):
        lines = cv2.HoughLinesP(
            frame,
            rho=6,
            theta=np.pi / 60,
            threshold=160,
            lines=np.array([]),
            minLineLength=40,
            maxLineGap=25
        )
        if lines is None : 
            return frame    
        else: 
            return lines    
        


class ShowImage(Pipeline):
    def map(self,frame):
        cv2.imshow('frame',frame)
        return frame

video_in_file = "test.mp4"

pipeline = ReadFrame(video_in_file) | ConvertToGray() | EdgeDetect() |  CropFrame() | DetectLine() | ShowImage()

for frame in pipeline:
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
