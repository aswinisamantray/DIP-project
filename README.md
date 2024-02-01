# Brightess Preserving Dynamic Histogram Equalization Technique(BPDHE)
Histogram Equalization method tends to introduce unnecessary visual deterioration such as the saturation effect. It is not useful in consumer electronics such as television because the method tends to produce undesirable artifacts. Changes the brightness of input image significantly making some of the regions of output image saturated with very bright or very dark intensities. BPDHE maintainS the original input brightness in the output image. 
It consists of five steps:
1.Smooth the histogram with Gaussian filter. 
2. Detection of the location of local maximums from the smoothed histogram.
3. Map each partition into a new dynamic range. 
4. Equalize each partition independently. 
5. Normalize the image brightness. 

