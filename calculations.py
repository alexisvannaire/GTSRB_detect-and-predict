


## libraries
import os
import pathlib
import shutil
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

## global variables
traffic_signs_names = [
    "speed limit 20", "speed limit 30", "speed limit 50", "speed limit 60", 
    "speed limit 70", "speed limit 80", "end speed limit 80", "speed limit 100", 
    "speed limit 120", "no overtaking",  "no overtaking by heavy goods vehicles",
    "crossroads minor road", "priority road", "give way", "stop", "no vehicles", 
    "no heavy goods vehicles", "no entry", "other danger", "curve left",
    "curve right", "series of curves", "uneven surface", "slippery surface", 
    "road narrows", "roadworks", "traffic signals", "pedestrians", "children", 
    "cyclists", "ice snow", "wild animals", "end all previously signed restrictions", 
    "turn right", "turn left", "go straight", "go straight or turn right", 
    "go straight or turn left", "keep right", "keep left", "roundabout",
    "end no overtaking", "end no overtaking by heavy goods vehicles"
]


##############################################################################################################
# Part 1 functions
### images dimensions
def get_images_dimensions(verbose=True):
    """
    Get gtsrb train images dimensions.

    Parameters
    ----------
    verbose: bool, default=True
        If True, information on the running state will be displayed.

    Returns
    -------
        pd.DataFrame
        A dataframe containing images dimensions (widths and heights)
    """
    # images location
    data_dir = "data/gtsrb/"
    train_data_dir = data_dir+"Train/"
    train_data_dir_path = pathlib.Path(train_data_dir)

    # loop on classes
    class_nb = len(os.listdir(train_data_dir))
    img_dimensions = []
    for i in range(class_nb):
        # images from current class
        class_i = list(train_data_dir_path.glob(str(i)+'/*'))
        # loop on current class images
        for j in range(len(class_i)):
            # height and width
            h, w, _ = cv2.imread(str(class_i[j])).shape
            img_dimensions.append([h, w])
        if verbose:
            print(f"\r{i+1}/{class_nb}  ", end="")
    # results into a dataframe
    img_dimensions_df = pd.DataFrame(img_dimensions, columns=["width", "height"])
    return img_dimensions_df

### images number per class
def get_images_number_per_class():
    """
    Compute the images number per class from the gtsrb train set.

    Returns
    -------
        pd.DataFrame
        A dataframe containing information on image number per class: "class_number", "class_name", "number".
    """
    global traffic_signs_names

    # data location
    train_data_dir = "data/gtsrb/Train/"
    class_folders = os.listdir(train_data_dir)

    # loop on classes
    class_nb = len(class_folders)
    images_number_per_class = []
    for i in range(class_nb):
        class_folder = class_folders[i]
        # images number
        images_number = len(os.listdir(train_data_dir+class_folder+"/"))
        images_number_per_class.append([int(class_folder), traffic_signs_names[int(class_folder)], images_number])
    # results into a dataframe
    images_number_per_class = pd.DataFrame(images_number_per_class, columns=["class_number", "class_name", "number"])
    images_number_per_class = images_number_per_class.sort_values(by="number", ascending=False).reset_index(drop=True)
    return images_number_per_class


##############################################################################################################
# Part 3 functions
### get predictions
def get_y_predictions(model, tf_dataset, class_names, verbose=False):
    """
    Get predictions from a model according to a dataset.

    Parameters
    ----------
    model: tensorflow model (classifier)
        A tensorflow model trained for the classification task on gtsrb.
    tf_dataset: tensorflow dataset
        A tensorflow dataset as we used to load train, val and test datasets.
    class_names: list
        A list of class names.
    verbose: bool, default=False
        If True, information on the running state will be displayed.
    
    Returns
    -------
        tuple
        A tuple of three objects:
            - a list of actual class names from a dataset
            - a list of numpy arrays corresponding to the predicted probabilities of classes per batch
            - a list of integers corresponding to the predicted class names as they can be found in the gtsrb dataset
    """
    ## loop on the tensorflow dataset batches
    cpt = 0
    y, y_probas, y_pred = [], [], []
    for features, labels in tf_dataset:
        y.extend(labels.numpy().tolist()) # actual class names
        X = features.numpy() # images
        y_proba = model.predict(X) # predicted probabilities
        y_probas.append(y_proba)
        y_pred.append(np.array(class_names)[y_proba.argmax(axis=1)]) # predicted class names
        cpt+=1
        if verbose:
            print("\rbatch n°"+str(cpt)+"        ", end="")
    if verbose:
        print()
    ## y_pred into a list of integers
    y_pred = list(map(lambda x: x.tolist(), y_pred))
    y_pred = sum(y_pred, [])
    y_pred = list(map(lambda x: int(x), y_pred))

    return y, y_probas, y_pred

### get confusion matrix
def get_confusion_matrix(y, y_pred):
    """
    Get confusion matrix from the tensorflow function.
    """
    return tf.math.confusion_matrix(y, y_pred).numpy()

### get wrong predictionsx
def get_wrong_predictions(y, y_pred, tf_ds, actual, predicted, batch_size):
    """
    Get wrong predictions according to the chosen actual and predicted ones.

    Parameters
    ----------
    y: list
        The list of actual images labels.
    y_pred: list
        The list of predicted images labels.
    tf_ds:
        A tensorflow dataset as we used to load train, val and test datasets.
    actual: int or str
        The name of an actual class.
    predicted: int or str
        The name of a predicted class.
    batch_size: int
        An integer corresponding to the batch size used for training.

    Returns
    -------
        list
        A list of images that have been confused given an actual and a predicted class.
    """
    ## get indexes of the chosen classes
    actual_i_indexes = np.where(np.array(y) == actual)[0]
    predicted_j_indexes = np.where(np.array(y_pred) == predicted)[0]
    indexes = np.intersect1d(actual_i_indexes, predicted_j_indexes)
    ## get corresponding images
    cpt = 0
    start, end = 0, batch_size-1
    error_images = []
    # loop on batches
    for features, _ in tf_ds:
        # find the right batch
        while(cpt < indexes.shape[0] and start <= indexes[cpt] and indexes[cpt] <= end):
            img_index = indexes[cpt] - start
            X = features.numpy()
            error_images.append(X[img_index,:,:,:]) # add the image into the list
            cpt += 1
        start += batch_size
        end += batch_size

    return error_images

#### get gtsrb test dataset as for train
def load_test_dataset_labels_and_create_folders(test_metadata, test_folderpath="", new_data_location="", delete=False):
    """
    Create subfolders for the Test folder as it is in the Train set of gtsrb.

    Parameters
    ----------
    test_metadata: pd.DataFrame
        A dataframe corresponding to the Test.csv file that is in the gtsrb dataset.
    test_folderpath: str, default=""
        The folderpath to the Test folder of gtsrb.
    new_data_location: str, default=""
        The folderpath where you want the class organized images to be saved.
    delete: bool, default=False
        If True, files will be moved. Otherwise they will just be copied and pasted.

    Returns
    -------
        list
        A list of actual labels of the created Test set.
    """
    ### check if the folder already exists
    if not(os.path.exists(new_data_location)):
        os.makedirs(new_data_location)
    else:
        ## check if the existing folder contains images
        if os.path.exists(os.path.join(new_data_location+"16/00000.png")):
            # get labels from the folder that has already been created
            labels = []
            for root, _, files in os.walk(test_folderpath):
                for name in files:
                    filepath = os.path.join(root, name)
                    label = test_metadata.loc[test_metadata["Path"] == "Test/"+name, "ClassId"].values[0]
                    labels.append(label)
            return labels
    ### if it doesn't exist, create subfolders corresponding to each class
    labels = sorted(test_metadata["ClassId"].unique())
    for label in labels:
        os.makedirs(new_data_location+str(label), exist_ok=True)
    ### get labels and copy/paste or move images
    labels = []
    ## loop on files
    for root, _, files in os.walk(test_folderpath):
        for name in files:
            # get the current label
            label = test_metadata.loc[test_metadata["Path"] == "Test/"+name, "ClassId"].values[0]
            labels.append(label)
            # the current image filepath and the new one
            filepath = os.path.join(root, name)
            new_filepath=new_data_location+str(label)+"/"+name
            # try to copy/paste or move the file
            try:
                # copy/paste
                shutil.copy2(filepath, new_filepath)
                # if you want to move it
                if delete:
                    # delete the original file
                    if pathlib.Path(filepath).suffix == ".png":
                        os.remove(filepath)
                        print(f'\r{name} moved to {new_filepath}    ', end="")
                else:
                    # doesn't delete
                    print(f'\r{name} copied to {new_filepath}    ', end="")
            except shutil.SameFileError:
                # in cas an error appears
                print('Error: nothing done with \r{name} on {new_filepath}')
        ## the base folder contains all the files, doesn't need to look further
        break
            
    return labels


