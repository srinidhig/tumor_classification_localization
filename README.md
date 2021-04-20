# Tumor Classification and Localization using Supervised Machine Learning
Part of Northeastern University DS5220 - Supervised Machine Learning Coursework

# Link to presentation: https://docs.google.com/presentation/d/16LyuDm7fGq9F2lS0T-F_uL7SyFkiyEypYcGcBrjDcHI/edit?usp=sharing

# Objective and significance:
MRI image is an important component in the detection of tumors, and plays a key role in identifying the tumors. Medical images do not need to invade the body tissues to clearly show the relevant lesions. Compared with anatomy, it has an irreplaceable advantage. Magnetic Resonance Imaging (Magnetic Resonance Imaging) proved an important literary genre in the early medical imaging community. A very important step for analyzing and diagnosing MRI images using a computer is to segment the images in advance.

# Problem Description:
The aim of this project is to explore the different methods used to classify MRI images in order to identify the tumors. Therefore, this study makes a major contribution to research on detection of tumors by demonstrating several algorithms used in classification of MRI images.

# Algorithms: 
Since this is a classification problem, we plan to start with more traditional algorithms such as KNN and SVM to classify the images into tumor / no tumor. 

KNN - chooses class based on distance from k neighbors. 

SVM - chooses a decision boundary based on maximum width or margin between the two classes. A support vector machine is trained by solving for a hyperplane that best separates the two classes in a given data set. When making a prediction, the SVM simply classifies the data according to the side of the hyperplane that the data is on, making the calculation simple once the SVM model is trained. The data given to the model are features that the implementers of the algorithm need to determine. These features could include shapes, colors, textures, etc.

Then, we plan to compare the results of these algorithms with deep learning techniques - primarily CNN, which is used for classifying images.

Apart from the classification of the MRI images themselves, we plan to go a step further and experiment with tumor localization, which will show the exact position of the tumor in the image by drawing a bounding box around it. The algorithms to try this out are YOLO and R-CNN. In recent years, CNN has gained traction for this specific task in the medical field and object localization is prevalent mostly in the development of AI / self-driving cars but is applicable in the medical field as well. 

# Dataset and Preprocessing Steps: 
The data can be obtained from https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection. The initial sample consisted of 253 MRI images, 155 of which belonged to the ‘yes’ label. 

https://figshare.com/articles/dataset/brain_tumor_dataset/1512427/5 This brain tumor dataset contains 3064 T1-weighted contrast-inhanced images with three kinds of brain tumor. Detailed information of the dataset can be found in readme file.

The preprocessing steps would primarily involve a deeper dive into the tumor vs. no-tumor images via image processing techniques such as edge detection, noise removal, etc. Rotational and translational steps would be added to add more data to the training set.

# Libraries and Tools:

Python: We plan to use Python3 for our code and the IDE will be Jupyter notebook or Google Colab. Below are the packages involved - 

Numpy: NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

OpenCV: OpenCV is a library of programming functions mainly aimed at real-time computer vision. Also useful for image processing.

Scikit-Learn: Scikit-learn is a free software machine learning library for the Python programming language. The KNN and SVM models will be built using this and will also be useful for feature extraction and model performance computation and comparison.

Tensorflow: TensorFlow is a free and open-source software library for machine learning. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks. The CNNs and Localization algorithms will be used from this library.

Keras: Keras is an open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library.

# Results: 
We will use the Training time and run time to compare different methods. In addition, confusion matrix summarizes the number of instances predicted correctly or incorrectly by a classification model and we will discuss the AUC curve and F1 score of the three models. For tumor localization, we would look at the accuracy of detecting images with tumors as well as minimizing error in localizing the tumors in these images.

# References:

You Only Look Once: Unified, Real-Time Object Detection, Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi. Link: https://arxiv.org/pdf/1506.02640.pdf

Convolutional Neural Networks by Andrew Ng (deeplearning.ai). Link: https://www.coursera.org/learn/convolutional-neural-networks

Classification of Brain Tumors from MRI Images Using a Convolutional Neural Network - MilicaM. Badža and Marko C. Barjaktarovic. Link: https://www.researchgate.net/publication/339994574_Classification_of_Brain_Tumors_from_MRI_Images_Using_a_Convolutional_Neural_Network

Dataset: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427/5

Machine Learning algorithms for Image Classification of hand digits and face recognition dataset by Tanmoy Das. Link: https://www.irjet.net/archives/V4/i12/IRJET-V4I12123.pdf
https://github.com/Abhishek-Arora/Image-Classification-Using-SVM 

MRI brain image classification using neural networks - Walaa Hussein Ibrahim; Ahmed AbdelRhman Ahmed Osman; Yusra Ibrahim Mohamed. Link: https://ieeexplore.ieee.org/abstract/document/6633943

Image Classification of Brain MRI Using Support Vector Machine - Noramalina Abdullah, Umi Kalthum Ngah, Shalihatun Azlin Aziz. Link: https://www.researchgate.net/profile/Shalihatun-Aziz-2/publication/252022435_Image_classification_of_brain_MRI_using_support_vector_machine/links/562462ba08aea35f26869042/Image-classification-of-brain-MRI-using-support-vector-machine.pdf
