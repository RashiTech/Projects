#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#identifying colors in an image


# In[1]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import argparse
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']


# In[4]:


#convert RGB to hex
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


# In[5]:


#method cvtColor allows us to convert the image rendering to a different color space. 
#by default, OpenCV reads image in the sequence Blue Green Red (BGR)
#To move from BGR color space to RGB, we use the method cv2.COLOR_BGR2RGB
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# In[10]:


def get_colors(image, number_of_colors, show_chart):
    '''
    Function to extract the top colors from the image and display them as a pie chart. I’ve named the method as get_colors and it takes 3 arguments:
    image: The image whose colors we wish to extract.
    number_of_colors: Total colors we want to extract.
    show_chart: A boolean that decides whether we show the pie chart or not.
    '''
    
    #Resize image to a smaller size but we do so to lessen the pixels which’ll reduce the time needed to extract the colors from the image.
    #KMeans expects the input to be of two dimensions, so we use Numpy’s reshape function to reshape the image data.
    
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    #KMeans algorithm creates clusters based on the supplied count of clusters. 
    #In our case, it will form clusters of colors and these clusters will be our top colors
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    
    return hex_colors, f'max color used {hex_colors[max(counts, key=counts.get)]}'


# In[11]:


get_colors(get_image('task.jpeg'), 8, True)


# In[ ]:


#Extracting text from image
#installing dependencies


# In[108]:


# Import required packages
import cv2
import pytesseract

# configurations
config = ('-l eng --oem 1 --psm 3')
# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Read image from which text needs to be extracted
img = cv2.imread("quotes.jpg")

# Preprocessing the image starts

# Convert the image to gray scale for proper contouring
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Performing OTSU threshold
#ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#ret, thresh2 = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10) 

# Specifying structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 2)

# Finding contours
#contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# Creating a copy of image
im2 = img.copy()

# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it

for i,cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)

    # Drawing a rectangle on copied image
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Cropping the text block for giving input to OCR
    cropped = im2[y:y + h, x:x + w]
    #plt.imshow(cropped)
    
    
    # Apply OCR on the cropped image
    text = pytesseract.image_to_string(cropped, config=config)
    print(i,text)
    

# For converting pdf file to jpeg format
# In[ ]:
!pip install pdf2image

# import module
from pdf2image import convert_from_path
 
# In[ ]: 
# Store Pdf with convert_from_path function
images = convert_from_path('Task1.pdf')
 
for i in range(len(images)):
   
      # Save pages as images in the pdf
    images[i].save('page'+ str(i) +'.jpg', 'JPEG')



