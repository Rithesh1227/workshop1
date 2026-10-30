# Workshop -1 Adding Sunglasses to Your Passport Photo Using OpenCV
# Name : Rithesh S
# Reg No : 212225220084
## Program
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

faceImage = cv2.imread('img2.jpeg')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")

faceImage.shape
#resized_faceImage.shape
faceImage.shape

# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread('sunglass.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")

# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(280,140))
print("image Dimension ={}".format(glassPNG.shape))

# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

# Make a copy
#faceWithGlassesNaive = resized_faceImage.copy()
faceWithGlassesNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithGlassesNaive[240:380,272:552]=glassBGR

plt.imshow(faceWithGlassesNaive[...,::-1])

# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask/255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
eyeROI= faceWithGlassesArithmetic[240:380,272:552]

# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI,(1-  glassMask ))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR,glassMask)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")

# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[240:380,272:552]=eyeRoiFinal

# Display the final result
plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses");
```
## Output
<img width="392" height="434" alt="download" src="https://github.com/user-attachments/assets/55d00d56-b400-4f22-96ba-ab0276672df5" />
<img width="565" height="434" alt="download" src="https://github.com/user-attachments/assets/105b913a-8aa8-43f3-8d6d-65acc06b735c" />
<img width="1219" height="329" alt="download" src="https://github.com/user-attachments/assets/efd6e269-aa6b-4377-8db5-befe7b0f82eb" />
<img width="392" height="418" alt="download" src="https://github.com/user-attachments/assets/9df17463-62a4-4a2e-a40e-445f17632b94" />
<img width="1606" height="292" alt="download" src="https://github.com/user-attachments/assets/9d5f5a3e-6fba-48c7-8c61-a693aa15bebc" />
<img width="1606" height="840" alt="download" src="https://github.com/user-attachments/assets/ac57e894-d539-4ee5-94ad-6fc3b419b4f9" />

## Result
Thus, adding sunglass to the passport photo using opencv is performed successfully.
