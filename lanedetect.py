
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import math

def region_of_interest(img,vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image



def draw_lines(img, lines, color=[0,0, 255], thickness=3):
    if lines is None:
        return
    
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img


def pipeline(image):

    height = image.shape[0]
    width = image.shape[1]


    region_of_interest_vertices = [
        (0,height),
        (width/2,height/2),
        (width,height)
    ]

    gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image,100,200)
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32)
    )

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
    if lines is None : return image


    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
            if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                continue
            if slope <= 0: # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])


    min_y = int(image.shape[0] * (3 / 5)) # <-- Just below the horizon
    max_y = image.shape[0] # <-- The bottom of the image
    if len(left_line_y) <= 0 : return image
    if len(left_line_x) <= 0 : return image
    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    if len(right_line_y) <= 0 : return image
    if len(right_line_x) <= 0 : return image    
    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))


    new_lines = [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]]

    line_image = draw_lines(
        image,
        lines = new_lines
    )

    return line_image


video_in_file = "test.mp4"
video_in = cv2.VideoCapture(video_in_file)
frame_width = int(video_in.get(3))
frame_height = int(video_in.get(4))


video_out_file = "test_out.mp4"
fourcc = cv2.VideoWriter.fourcc(*"MJPG")
video_out = cv2.VideoWriter(video_out_file, fourcc, 10, (frame_width,frame_height))

while (video_in.isOpened()):
    ret, frame = video_in.read()
    if ret == True:
        video_out.write(pipeline(frame))
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
video_out.release()
