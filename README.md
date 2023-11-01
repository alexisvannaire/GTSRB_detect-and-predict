Traffic signs detection and classification based on GTSRB dataset
=================================

This project aims at showing how to build AI models based on the GTSRB dataset.

It will help you to get started with the Computer Vision field. You will find all the functions I used for processing data, handling models, visualizations, and so on.

You can find the associated blog post on Medium here: [...](https://www.google.com)

## The project

Given an image of the road, the goal is to detect traffic signs and predict which ones they are.

So you will see how to build an object detection model with the Detecto package, how to label images in order to train it, how to train CNNs and MobileNets on the GTSRB dataset with Tensorflow, how to analyze and visualize predictions.

You will find notebooks and python files.
Notebooks for visualizing the process, analysis, concrete examples and python files for functions.

You can begin by looking at the notebooks that correspond to:
 * n°1 - **Introduction and Datasets**
 * n°2 - **Object Detection** 
 * n°3 - **Classifiction**
 * n°4 - **Full Process**

## Requirements

If you want to run the notebooks you will need:

* to download the GTSRB dataset, you can find it here: [https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) 

* to have or install the following packages: 
    - `tensorflow`
    - `detecto` ([https://github.com/alankbi/detecto](https://github.com/alankbi/detecto)) 
    - `labelImg`
    - `sklearn`
    - `cv2`
    - `PIL`
    - `plotly`
    - `ipython`
    - `random`
    - `matplotlib`
    - `pandas`
    - `numpy`
    - `shutil`
    - `pathlib`
    - `xml`
    - `json`
    - `functools`
    - `os`

<br> 

You will find some details at the begining of the notebook n°1: `1_detect-and-predict`, and if needed, in the `requirements.txt` file.



