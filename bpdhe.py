
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

def gaussian_filter(shape=(1,9),sigma=1.0762):
    m,n = [(i-1.)/2. for i in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1] #Arrays y and x have positions in y and x coordinates respectively
    h = np.exp( -(x**2 + y**2) / (2.*sigma**2) ) #Gaussian filter function
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0 #Remove the very small values
    sumh = h.sum()
    if sumh != 0:
        h /= sumh #Normalize the values of the array h
    return h
  
def map_ranges(sub_hist,index,mini,rangee,start,end,span,factor,M,hist_i):
  for m in range(0,len(index)-1):
    sub_hist.append( np.array(hist_i[0][index[m]:index[m+1]]) ) 
    M[m] = np.sum(sub_hist[m])
    low = mini + index[m]
    high = mini + index[m+1] - 1
    span[m] = high - low + 1
    factor[m] = span[m] * np.log10(M[m])
    factor_sum = np.sum(factor)
  for m in range(0,len(index)-1):
    rangee[m] = np.round((256-mini)*factor[m]/factor_sum)
  start[0] = mini
  end[0] = mini + rangee[0]-1
  for m in range(1,len(index)-1):
    start[m] = start[m-1] + rangee[m-1]
    end[m] = end[m-1] + rangee[m]

def bpdhe(image):
  image = image.astype('uint8')
  hue_sat_val = cv.cvtColor(image, cv.COLOR_RGB2HSV) #Change the color space from RGB to HSV
  hue,sat,val = cv.split(hue_sat_val)
  hue=(hue/255.0)
  sat=(sat/255.0)

  maxi = np.max(val)
  mini = np.min(val)
  bins = (maxi-mini) + 1
  hist_i = np.histogram(val,bins=bins)
  hist_i = hist_i[0].reshape(1,len(hist_i[0]))

  gauss_filter = gaussian_filter()
  blur_hist = cv.filter2D(hist_i.astype('float32'),-1,gauss_filter, borderType=cv.BORDER_REPLICATE)
  derivFilter = np.array([[-1,1]])
  slope_hist =  cv.filter2D(blur_hist.astype('float32'),-1,derivFilter, borderType=cv.BORDER_REPLICATE)
  sign_hist = np.sign(slope_hist)
  meanFilter = np.array([[1/3,1/3,1/3]])
  smooth_sign_hist =  np.sign(cv.filter2D(sign_hist.astype('float32'),-1,meanFilter, borderType=cv.BORDER_REPLICATE))
  cmpFilter = np.array([[1,1,1,-1,-1,-1,-1,-1]])
  p = 1
  index = [0]
  for n in range(0,bins-7):
    C = (smooth_sign_hist[0][n:n+8] == cmpFilter)*1
    if np.sum(C) == 8.0:
      p+=1
      index.append(n+3)

  index.append(bins)

  factor = np.zeros(shape=(len(index)-1,1))
  span = factor.copy()
  M = factor.copy()
  rangee = factor.copy()
  start = factor.copy()
  end = factor.copy()
  sub_hist = []
  map_ranges(sub_hist,index,mini,rangee,start,end,span,factor,M,hist_i)

  y = []
  equalization_mapping = np.zeros(shape=(1,mini))
  equalization_mapping = equalization_mapping.tolist()
  equalization_mapping = (equalization_mapping[0])
  for m in range(0, len(index)-1):
    hist_cum = np.cumsum(sub_hist[m]) 
    c = hist_cum/M[m]
    y.append(np.array(np.round(start[m]+(end[m]-start[m])*c)))
    x = y[m].tolist()
    equalization_mapping=equalization_mapping+x
  
  i_s = np.zeros(shape=val.shape)
  for n in range(mini,maxi+1):
    lc = (val== n)
    i_s[lc] = (equalization_mapping[n])/255
  hsi_0 = cv.merge([hue, sat, i_s])
  hsi_0 = (hsi_0 * 255).astype('uint8')
  final = cv.cvtColor(hsi_0, cv.COLOR_HSV2RGB)
  
  Mi=np.mean(image)
  Mo=np.mean(final)
  factor=(Mi/Mo)
  out_img=Image.fromarray(np.multiply(image, factor).astype(np.uint8))  
  
  return out_img

input_image = cv.imread('DIP_image.jpg')
enhanced_image = bpdhe(input_image)
plot_hist(get_histogram(input_image),"Input image histogram")
plot_hist(get_histogram(enhanced_image),"Output image histogram")

M_input_image=np.mean(input_image)
M_enhanced_image=np.mean(enhanced_image)
print(M_enhanced_image-M_input_image)


# Display the result image
plt.imshow(enhanced_image)
plt.title('Enhanced Image')
plt.axis('off')

plt.show()
