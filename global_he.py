import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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

def global_he(input_img):
    width, height = input_img.shape[:2]

    # put pixels in a 1D array by flattening out img array
    flat = input_img.flatten()

    #getting histogram
    hist = get_histogram(flat)
    MN = width*height
    #finding the probability
    array_pdf = hist/MN

    #setting the curriculum density
    CDF = 0
    CDF_matrix = np.zeros(256)
    for i in range(1, 256):
        CDF = CDF + array_pdf[i]
        CDF_matrix[i] = CDF
    
    final_array = np.zeros(256)
    final_array = (CDF_matrix * 255)
    for i in range (1,256):
        final_array[i] = np.round(final_array[i])

    img_new = final_array[flat]

    # put array back into original shape since it was flattened 
    img_new = np.reshape(img_new, input_img.shape)
    return img_new
    
current_directory = os.getcwd()
aambe=[]
for filename in os.listdir(current_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
            input_image = np.asarray(Image.open(os.path.join(current_directory, filename)))
            enhanced_image = global_he(input_image)
            M_input_image=np.mean(input_image)
            M_enhanced_image=np.mean(enhanced_image)
            aambe.append(np.abs(M_input_image-M_enhanced_image))

print(np.mean(aambe))

# # Display the result image
# plt.imshow(enhanced_image)
# plt.title('Enhanced Image')
# plt.axis('off')
# plt.show()
