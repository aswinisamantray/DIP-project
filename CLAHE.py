import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_hist(histogram,title):
    plt.bar(np.arange(256), histogram, width=1, color='gray')
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
    
def get_histogram(img):
    img = np.clip(img, 0, 255).astype(int)
    histogram = np.zeros(256)
    # loop through pixels and sum up counts of pixels
    for pixel in img:
        histogram[pixel] += 1
    # return our final result
    return histogram

def dynamic_histogram_equalization(img, clip_limit=3.0, grid_size=(8, 8)):
    lab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv.split(lab_img)

    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl_channel = clahe.apply(l_channel)

    enhanced_lab = cv.merge([cl_channel, a_channel, b_channel])
    enhanced_img = cv.cvtColor(enhanced_lab, cv.COLOR_LAB2RGB)

    return enhanced_img

current_directory = os.getcwd()
aambe=[]
for filename in os.listdir(current_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
            input_image = cv.imread(os.path.join(current_directory, filename))
            enhanced_image = dynamic_histogram_equalization(input_image)
            M_input_image=np.mean(input_image)
            M_enhanced_image=np.mean(enhanced_image)
            aambe.append(np.abs(M_input_image-M_enhanced_image))

print(np.mean(aambe))

# plot_hist(get_histogram(input_image),"Input image histogram")
# plot_hist(get_histogram(enhanced_image),"Output image histogram")
# plt.imshow(enhanced_image)
# plt.title('Enhanced Image')
# plt.axis('off')

# plt.show()
