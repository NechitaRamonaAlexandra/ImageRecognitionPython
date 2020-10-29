# ImageRecognitionPython

## Overview
This project will make use the tensorflow framework, which shall help with the creation and training of a neural network.
Our team works with convolutional neural networks which are classes of deep neural networks usually applied to analyzing visual imagery.
The main purpose of the project is creating a system which can recognize and classify images with various animals.

The project shall be able to receive as input an image and detect if it contains an animal, and if it does, to determine what animal it is. Initially the system will be separating them by classes, family (being able to see it’s a feline, but not knowing yet if it is a cat or a tiger), and upon further development distinguishing even certain breeds.

## Implementation
One type of image recognition algorithm is an image classifier. It takes an image (or part of an image) as an input and predicts what the image contains. The output is a class label, such as dog, cat or amphibian. The algorithm needs to be trained to learn and distinguish between classes.
In a simple case, to create a classification algorithm that can identify images with dogs, you’ll train a neural network with thousands of images of dogs, and thousands of images of backgrounds without dogs. The algorithm will learn to extract the features that identify a “dog” object and correctly classify images that contain dogs. 
Image segmentation is a critical process in computer vision. It involves dividing a visual input into segments to simplify image analysis. Segments represent objects or parts of objects, and comprise sets of pixels, or “super-pixels”. Image segmentation sorts pixels into larger components, eliminating the need to consider individual pixels as units of observation. There are three levels of image analysis:
-	Object Classification
-	Object Detection
-	Segmentation
We will use python in order to implement such algorithm.

TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks.

## Statistics of results
After each epoch, we printed the loss and accuracy values, and as it can be seen, the accuracy increases every time.  
Average val acc: 0.32  
Average val loss: 2.87  

![Image](https://github.com/NechitaRamonaAlexandra/ImageRecognitionPython/blob/main/res.png)

**Graphs**

![Image](https://github.com/NechitaRamonaAlexandra/ImageRecognitionPython/blob/main/Plot.png)


![Image](https://github.com/NechitaRamonaAlexandra/ImageRecognitionPython/blob/main/Figure_12.png)
