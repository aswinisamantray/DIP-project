# Brightess Preserving Dynamic Histogram Equalization Technique(BPDHE)
Histogram Equalization method tends to introduce unnecessary visual deterioration such as the saturation effect. It is not useful in consumer electronics such as television because the method tends to produce undesirable artifacts. Changes the brightness of input image significantly making some of the regions of output image saturated with very bright or very dark intensities. BPDHE maintains the original input brightness in the output image. <br>
## It consists of five steps:<br>
1.Smooth the histogram with Gaussian filter. <br>
2. Detection of the location of local maximums from the smoothed histogram. <br>
3. Map each partition into a new dynamic range. <br>
4. Equalize each partition independently. <br>
5. Normalize the image brightness. <br>

In this project three histogram equalization techinques are implemented namely BPDHE, CLAHE, Histogram Equalization. Applying the three techniques on a set of images and then calculating Average Absolute Mean Brightness Error shows that the error value for BPDHE is the lowest.
