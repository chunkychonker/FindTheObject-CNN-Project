# FindTheObject-CNN-Project-

Project plan 

V1 (main project): “sign in the wild” dataset

In V1, i'm going to generate my own training images to keep the problem simple and controlled:
	•	start with real traffic sign images (cropped signs) (get from GTSRB)
	•	paste one sign onto a larger background image at a random location
	•	add randomness (brightness changes, blur/noise, small occlusions) to make it more realistic
	•	create a matching target heatmap that marks where the pasted sign is

This makes the “find the sign anywhere in the image” idea very clear

V2 (stretch goal): Test on real road images

V2: plan to test the model on real driving/road images (where signs can be smaller, lighting is harder, and there might be multiple signs)

What the model outputs - 
  Given an RGB image, the network predicts:
  	•	Class prediction: which traffic sign it thinks it is
  	•	Heatmap prediction: where the sign is (the brightest spot should land on the sign)


Key CNN ideas this project practices:
	•	Convolution layers (learning features from small local regions)
	•	Padding / stride / pooling (and how they change feature map sizes and localization quality)
	•	Training a model with two outputs (classification + localization)
	•	Data augmentation to reduce overfitting (color changes, blur/noise, occlusion, etc.)
	•	(Optional) ResNet-style skip connections as an upgrade to the baseline CNN


Evaluation:
  Classification
	  •	Accuracy
	  •	Confusion matrix
  Localization (heatmap)
	  •	Pointing accuracy: is the heatmap’s highest point inside the true sign region?
	  •	Center error: how far the predicted point is from the true sign center (in pixels)
	  •	(Optional) turn the heatmap into a box and compute IoU

Demo / visuals
  The repo includes visualizations that show:
	  •	the input image
	  •	the predicted class
	  •	the predicted heatmap overlay (so you can see what the model is “looking at”)
  (Optional: add a small Streamlit app to upload an image and see results.)
