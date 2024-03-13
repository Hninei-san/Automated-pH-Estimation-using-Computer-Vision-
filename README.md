# Automated-pH-Estimation-using-Computer-Vision-
Automated pH Estimation using Computer Vision  for Water Quality Assessment
Members :
1. 65011289 Cherry Hlaing Kyaw
2. 65011303 Hnin Ei San
3. 65011497 Kyi Khaing Khant Soe

To develop a computer vision system capable of estimating the pH level of water samples by analyzing color changes on test strips
1. Automation: Traditional pH measurement methods require manual observation and interpretation. By using computer vision, we automate the process, reducing the need for manual intervention.
2. Visualization: Provide users with a clear understanding of the pH distribution within the sample, facilitating further analysis.
3. Accessibility:  Make pH estimation more accessible to a wider range of users. Only with the captured images, users can easily implement and utilize the system for pH analysis.

We utilize computer vision techniques and machine learning algorithms to analyze test strip images and determine pH values.The code has three main parts: image preprocessing, color extraction, and pH level classification.

In the preprocessing stage, we convert the input image to grayscale, apply Gaussian blur, and perform edge detection using the Sobel operator. After binary thresholding to convert the resulting image into black and white, we use morphological opening operations to remove any small dots and detect lines from the image using the Hough transform. We then define the area containing all the lines as our region of interest (ROI) and extract it.

In the color extraction stage, we perform K-means clustering to find dominant colors within the ROI. 
In the pH level classification stage, we use a K-nearest neighbors classifier to predict the pH value based on the colors detected in the ROI.
