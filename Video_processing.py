#%%
# EXTRACTING FRAMES FROM THE GIVEN VIDEO
import cv2
 
#%%
# Opens the Video file
cap= cv2.VideoCapture('C:/Users/Ayush/Desktop/Road_lanes/video.mp4')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('C:/Users/Ayush/Desktop/Road_lanes/Frame_test/Frame'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()


#%%
# PROCESSING FRAME_TEST_1

# %%
try:
    import os
    import re
    import cv2
    import numpy as np
    from tqdm import tqdm_notebook
    import matplotlib.pyplot as plt
    print("Libraries Imported")
except:
    print("Error in Importing Libraries")

# %%
frames_path = os.listdir('C:/Users/Ayush/Desktop/Road_lanes/Frame_test_1')
frames_path.sort(key=lambda f: int(re.sub('\D', '', f)))

# %%
# load frames
images_list=[]
for i in tqdm_notebook(frames_path):
    img = cv2.imread('C:/Users/Ayush/Desktop/Road_lanes/Frame_test_1/'+i)
    images_list.append(img)

# %%
import gc
gc.collect()   

# %%
# specify frame index
indx = 100

# plot frame
plt.figure(figsize=(10,10))
plt.imshow(images_list[indx][:,:,0], cmap= "gray")
plt.show()

# %%

# creating a zero array
stencil = np.zeros_like(images_list[indx][:,:,0])

# specifying the coordinates of the polygon
polygon = np.array([[0,900], [1710,0], [1990,300], [1300,1510]])

# fill polygon with ones
cv2.fillConvexPoly(stencil, polygon, 1)

# %%
# plot polygon
plt.figure(figsize=(10,10))
plt.imshow(stencil, cmap= "gray")
plt.show()

# %%
# apply polygon as a mask on the frame
img = cv2.bitwise_and(images_list[indx][:,:,0], images_list[indx][:,:,0], mask=stencil)

# plot masked frame
plt.figure(figsize=(10,10))
plt.imshow(img, cmap= "gray")
plt.show()

# %%
# applying image thresholding
ret, thresh = cv2.threshold(img, 130, 145, cv2.THRESH_BINARY)

# plot image
plt.figure(figsize=(10,10))
plt.imshow(thresh, cmap= "gray")
plt.show()

# %%
lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=0)

# create a copy of the original frame
dmy = images_list[indx][:,:,0].copy()

# draw Hough lines
for line in lines:
  x1, y1, x2, y2 = line[0]
  cv2.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3)

# plot frame
plt.figure(figsize=(10,10))
plt.imshow(dmy, cmap= "gray")
plt.show()

# %%
temp = 0

for img in tqdm_notebook(images_list):
  
  # apply frame mask
  masked = cv2.bitwise_and(img[:,:,0], img[:,:,0], mask=stencil)
  
  # apply image thresholding
  ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)

  # apply Hough Line Transformation
  lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=0)
  dmy = img.copy()
  
  # Plot detected lines
  try:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      cv2.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3)
  
    cv2.imwrite('C:/Users/Ayush/Desktop/Road_lanes/detected_test/'+str(temp)+'.png',dmy)
  
  except TypeError: 
    cv2.imwrite('C:/Users/Ayush/Desktop/Road_lanes/detected_test/'+str(temp)+'.png',img)

  temp+= 1


# %%
# PROCESSING FRAME_TEST_2

#%%
frames_path = os.listdir('C:/Users/Ayush/Desktop/Road_lanes/Frame_test_2')
frames_path.sort(key=lambda f: int(re.sub('\D', '', f)))

# %%
# load frames
images_list=[]
for i in tqdm_notebook(frames_path):
    img = cv2.imread('C:/Users/Ayush/Desktop/Road_lanes/Frame_test_2/'+i)
    images_list.append(img)

# %%
temp = 1328

for img in tqdm_notebook(images_list):
  
  # apply frame mask
  masked = cv2.bitwise_and(img[:,:,0], img[:,:,0], mask=stencil)
  
  # apply image thresholding
  ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)

  # apply Hough Line Transformation
  lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=0)
  dmy = img.copy()
  
  # Plot detected lines
  try:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      cv2.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3)
  
    cv2.imwrite('C:/Users/Ayush/Desktop/Road_lanes/detected_test/'+str(temp)+'.png',dmy)
  
  except TypeError: 
    cv2.imwrite('C:/Users/Ayush/Desktop/Road_lanes/detected_test/'+str(temp)+'.png',img)

  temp+= 1


# %%
# PROCESSING FRAME_TEST_3

#%%
frames_path = os.listdir('C:/Users/Ayush/Desktop/Road_lanes/Frame_test_3')
frames_path.sort(key=lambda f: int(re.sub('\D', '', f)))

# %%
# load frames
images_list=[]
for i in tqdm_notebook(frames_path):
    img = cv2.imread('C:/Users/Ayush/Desktop/Road_lanes/Frame_test_3/'+i)
    images_list.append(img)

# %%
temp = 2332

for img in tqdm_notebook(images_list):
  
  # apply frame mask
  masked = cv2.bitwise_and(img[:,:,0], img[:,:,0], mask=stencil)
  
  # apply image thresholding
  ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)

  # apply Hough Line Transformation
  lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 30, maxLineGap=0)
  dmy = img.copy()
  
  # Plot detected lines
  try:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      cv2.line(dmy, (x1, y1), (x2, y2), (255, 0, 0), 3)
  
    cv2.imwrite('C:/Users/Ayush/Desktop/Road_lanes/detected_test/'+str(temp)+'.png',dmy)
  
  except TypeError: 
    cv2.imwrite('C:/Users/Ayush/Desktop/Road_lanes/detected_test/'+str(temp)+'.png',img)

  temp+= 1


# %%
# BACK TO VIDEO FROM IMAGES/FRAMES
# input frames path
pathIn1= 'C:/Users/Ayush/Desktop/Road_lanes/detected_test_1/'
pathIn2= 'C:/Users/Ayush/Desktop/Road_lanes/detected_test_2/'
pathIn3= 'C:/Users/Ayush/Desktop/Road_lanes/detected_test_3/'

# output path to save the video
pathOut1 = 'C:/Users/Ayush/Desktop/Road_lanes/output_video_1.mp4'
pathOut2 = 'C:/Users/Ayush/Desktop/Road_lanes/output_video_2.mp4'
pathOut3 = 'C:/Users/Ayush/Desktop/Road_lanes/output_video_3.mp4'

# specify frames per second
fps = 30.0

# %%
from os.path import isfile, join

# get file names of the frames
files1 = [f for f in os.listdir(pathIn1) if isfile(join(pathIn1, f))]
files1.sort(key=lambda f: int(re.sub('\D', '', f)))

files2 = [f for f in os.listdir(pathIn2) if isfile(join(pathIn2, f))]
files2.sort(key=lambda f: int(re.sub('\D', '', f)))

files3 = [f for f in os.listdir(pathIn3) if isfile(join(pathIn3, f))]
files3.sort(key=lambda f: int(re.sub('\D', '', f)))

# %%
frame_list_1 = []

for i in tqdm_notebook(range(len(files1))):
    filename=pathIn1 + files1[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_list_1.append(img)

#%%
# write the video
out = cv2.VideoWriter(pathOut1,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_list_1)):
    # writing to a image array
    out.write(frame_list_1[i])

out.release()

# %%
frame_list_2 = []

for i in tqdm_notebook(range(len(files2))):
    filename=pathIn2 + files2[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_list_2.append(img)

# %%
# write the video
out = cv2.VideoWriter(pathOut2,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_list_2)):
    # writing to a image array
    out.write(frame_list_2[i])

out.release()

# %%
frame_list_3 = []

for i in tqdm_notebook(range(len(files3))):
    filename=pathIn3 + files3[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_list_3.append(img)

# %%
# write the video
out = cv2.VideoWriter(pathOut3,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_list_3)):
    # writing to a image array
    out.write(frame_list_3[i])

out.release()

#%%
# MERGING ALL 3 PROCESSED VIDEO

# %%
from moviepy.editor import VideoFileClip, concatenate_videoclips

#%%
os.chdir("C:/Users/Ayush/Desktop/Road_lanes")
os.listdir()

# %%
video_1 = VideoFileClip("output_video_1.mp4")
video_2 = VideoFileClip("output_video_2.mp4")
video_3 = VideoFileClip("output_video_3.mp4")

# %%
final_video= concatenate_videoclips([video_1, video_2, video_3])
final_video.write_videofile("final_output_video.mp4")

