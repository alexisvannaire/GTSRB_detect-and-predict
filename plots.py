


## libraries
import os
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf

## other libraries
import calculations

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
### add breakline on a too long title
def breakline_titles(title):
    """
    Add breakline to a title to make it more readable.
    """
    threshold = 15
    if len(title) > threshold:
        base = title.split(": ")[0]+": "
        str_length = len(base)
        words = title.split(": ")[1].split("-")
        cpt = 0
        while(str_length < threshold):
            if cpt == 0:
                new_base = base+words[cpt]
            else:
                new_base = base+("-".join(words[:cpt]))
            str_length = len(new_base)
            cpt += 1
        title = "\n"+new_base+"\n"+("-".join(words[cpt:]))
    else:
         title = "\n"+title+"\n"
    return " ".join(title.split("-"))

### wikipedia traffic signs
def plot_wiki_traffic_signs_classes(ncols=8, title_size=18):
    """
    Plot wikipedia traffic signs used in the GTSRB dataset.

    Parameters
    ----------
    ncols: int, default=8
        Column number you want for the plot.
    title_size: int, default=18
        Font size of subtitles.

    Return
    ------
        Nothing is returned.
    """
    # files
    wiki_imgs_folderpath = "data/wikipedia/"
    wiki_imgs_filenames = os.listdir(wiki_imgs_folderpath)
    n_wiki_imgs = len(wiki_imgs_filenames)
    # sort files in the right order
    wiki_imgs_filenames_indexes = np.array(list(map(lambda x: int(x.split("_")[0]), wiki_imgs_filenames)))
    arr = np.array([
        (wiki_imgs_filenames_indexes[i], wiki_imgs_filenames[i]) for i in range(n_wiki_imgs)
    ], dtype=[('x', int), ('y', 'U100')])
    arr.sort(order='x')
    ordered_filenames = arr["y"].tolist()
    # manage columns number
    if ncols > n_wiki_imgs:
        m = 8
    else:
        m = ncols
    # loop on subplots
    n = n_wiki_imgs//m + 1
    fig, axs = plt.subplots(nrows=n, ncols=m, figsize=(m*3, n*3))
    for i in range(n_wiki_imgs):
        # read image
        filename = ordered_filenames[i]
        img = cv2.imread(wiki_imgs_folderpath+filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        # plot
        row, col = i//m, i%m
        axs[row, col].imshow(img)
        # title
        title = f"{i}: {filename.split('.')[0].split('_')[1]}"
        title = breakline_titles(title)
        axs[row, col].set_title(title, size=title_size)
        # remove axis
        axs[row, col].axis("off")
    # the remaining spaces
    for i in range(n_wiki_imgs, n*m):
        row, col = i//m, i%m
        axs[row, col].axis("off")

    plt.tight_layout()
    plt.show()

### train gtsrb dataset random traffic signs
def plot_dataset_traffic_signs_classes(ncols=8, title_size=18, gtsrb_exists=True):
    """
    Plot dataset traffic signs from the train set of the GTSRB dataset.

    Parameters
    ----------
    ncols: int, default=8
        Column number you want for the plot.
    title_size: int, default=18
        Font size of subtitles.
    gtsrb_exists: bool, default=True
        If True, images will be taken from the dataset (has to be stored on "data/gtsrb/Train/"). Otherwise the image from the "imgs" will just be displayed.

    Return
    ------
        Nothing is returned.
    """
    
    global traffic_signs_names

    if not(gtsrb_exists) or not(os.path.exists("data/gtsrb/Train/")):
        return Image.open("imgs/plot_dataset_traffic_signs_classes_example.png").convert("RGB")
    else:
        # filepath
        gtsrb_folderpath = "data/gtsrb/Train/"
        gtsrb_folderpath_classes = os.listdir(gtsrb_folderpath)
        # sort
        gtsrb_folderpath_classes = sorted(list(map(lambda x: int(x), gtsrb_folderpath_classes)))
        n_classes = len(gtsrb_folderpath_classes)
        # manage columns
        if ncols > n_classes:
            m = 8
        else:
            m = ncols
        # loop on subplots
        n = n_classes//m+1
        fig, axs = plt.subplots(nrows=n, ncols=m, figsize=(m*3, n*3))
        for i in range(n_classes):
            # filepath
            folderpath = f"{gtsrb_folderpath}{gtsrb_folderpath_classes[i]}/"
            filenames = os.listdir(folderpath)
            filename = filenames[np.random.randint(len(filenames))]
            # read
            img = cv2.imread(folderpath+filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # plot
            row, col = i//m, i%m
            axs[row, col].imshow(img)
            # title
            title = f"{i}: {'-'.join(traffic_signs_names[i].split(' '))}"
            title = breakline_titles(title)
            axs[row, col].set_title(title, size=title_size)
            # remove axis
            axs[row, col].axis("off")
        # the remaining spaces
        for i in range(n_classes, n*m):
            row, col = i//m, i%m
            axs[row, col].axis("off")

        plt.tight_layout()
        plt.show()

### images dimensions distribution
def plot_images_dimensions_distribution(gtsrb_exists=True, verbose=True):
    """
    Plot the images widths and heights distribution.

    Parameters
    ----------
    gtsrb_exists: bool, default=True
        If True, images will be taken from the dataset (has to be stored on "data/gtsrb/Train/"). Otherwise the image from the "imgs" will just be displayed.

    verbose: bool, default=True
        If True, state of the process will be displayed. Otherwise it won't.

    Return
    ------
        tuple of a:
            - PIL.Image or plotly.graph_objects.Figure
                The images dimensions ditribution plot.
            - pandas.DataFrame
                The images dimensions data.
        
    """
    if not(gtsrb_exists) or not(os.path.exists("data/gtsrb/Train/")):
        return Image.open("imgs/img_dimensions_example.png").convert("RGB"), None
    else:
        # get images dimensions
        img_dimensions = calculations.get_images_dimensions(verbose=verbose)
        # plot
        fig = px.scatter(
            img_dimensions, x="width", y="height", opacity=0.1,
            marginal_x="histogram", marginal_y="histogram",
            title="Training set image dimensions")
        fig.update_layout(template="plotly_white")
        return fig, img_dimensions

### images dimensions distribution
def plot_images_number_per_class(gtsrb_exists=True):
    """
    Plot the images number per class.

    Parameters
    ----------
    gtsrb_exists: bool, default=True
        If True, images will be taken from the dataset (has to be stored on "data/gtsrb/Train/"). Otherwise the image from the "imgs" will just be displayed.

    verbose: bool, default=True
        If True, state of the process will be displayed. Otherwise it won't.

    Return
    ------
        tuple of a:
            - PIL.Image or plotly.graph_objects.Figure
                The images number per class bar plot.
            - pandas.DataFrame
                The images number per class data.
        
    """
    if not(gtsrb_exists) or not(os.path.exists("data/gtsrb/Train/")):
        return Image.open("imgs/images_number_per_class.png").convert("RGB"), None
    else:
        # get images number per class
        images_number_per_class = calculations.get_images_number_per_class()
        # plot
        fig = px.bar(
            images_number_per_class, x='class_name', y='number',
            title="Training set class sizes")
        fig.update_layout(
            template="plotly_white", xaxis_tickangle=42,
            xaxis_title="Class", yaxis_title="Number")
        fig.update_layout(template="plotly_white")
        return fig, images_number_per_class

### make a gif from images
def make_gif(folderpath, output_filepath, step=1, duration=334, loop=0, frames_names=None):
    """
    Make a gif from images.

    Parameters
    ----------
    folderpath: str
        The folderpath to images.
    output_filepath: str
        The filepath of the gif. 
    step: int, default=1
        An integer giving the step to apply to frames_names (filenames).
        If step=3, one image in three will be taken.
    duration: int, default=334
        An integer corresponding to the duration for each frame in milliseconds.
    loop: int, default=0
        An integer corresponding to the loop count, 0 for infinite loop.
    frames_names: list or None, default=None
        A list of image file names. Note that the order will be kept.
        If None, they will be taken using 'os.listdir(folderpath)'. 
        Note that they may be ordered differently from the way you would have wanted.

    Returns
    -------
        Nothing is returned.

    Example
    -------

    If you want to create a gif from images named as: "frame0.png", "frame1.png", ..., "frame99.png". You just have to create frames_names this way:

    frames_names = [f"frame{i}.jpg" for i in range(len(os.listdir(folderpath)))]

    And then, you can give them to the function:

    make_gif(folderpath, output_filepath, step=1, duration=334, loop=0, frames_names=frames_names)

    """
    # get filenames
    if frames_names is None:
        frames_names = os.listdir(folderpath)
    # read images
    images = [Image.open(folderpath+frames_names[i]) for i in range(0, len(frames_names), step)]
    # create folderpah if needed
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    # make gif
    images[0].save(output_filepath,
               save_all=True,
               append_images=images[1:],
               duration=duration,  # duration for each frame in milliseconds
               loop=loop)  # loop count, 0 for infinite loop


##############################################################################################################
# Part 2 functions
### plot labeled images with object detection predictions
def labeled_image_with_object_detection_predictions(img, od_model, classes, threshold, filepath):
    """
    Save a labeled image with predictions from object detection model.

    Parameters
    ----------
    img: numpy.array
        The numpy array you would get by reading an image this way:
        np.array(Image.open(filepath))
    od_model: detecto-like model
        An object detection model as the one created with the detecto package that can be used this way:
        labels, boxes, scores = od_model.predict(img)
    classes: list
        A list of classes you want keep to display predictions.
    threshold: float
        A probability threshold you want to apply to filter predictions. 
    filepath: str
        The filepath where you want the new image to be saved.

    Returns
    -------
        Nothing is returned.

    """
    ### get the image predictions
    labels, boxes, scores = od_model.predict(img)
    
    ### manage filters
    if len(labels) > 0:
        # convert tensor into numpy arrays
        boxes = boxes.numpy()
        scores = scores.numpy()
        # get classes filter indexes
        indexes = []
        for i in range(len(classes)):
            indexes.extend(np.where(np.array(labels) == classes[i])[0].tolist())
        indexes = np.array(indexes)
        if indexes.shape[0] > 0:
            labels, boxes, scores = np.array(labels)[indexes].tolist(), boxes[indexes], scores[indexes]
            # get predictions probability filter indexes
            indexes = np.where(scores >= threshold)[0]
            if indexes.shape[0] > 0:
                labels, boxes, scores = np.array(labels)[indexes].tolist(), boxes[indexes], scores[indexes]
            else:
                labels, boxes, scores = None, None, None
        else:
                labels, boxes, scores = None, None, None
    else:
        labels, boxes, scores = None, None, None

    ### plot
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig, ax = plt.subplots() 
    ax.imshow(img)
    center_width = img.shape[1]//2
    ## if there is no selected predictions, it'll just save the original image
    if boxes is None:
        pass
    else:
        # manage all the predictions
        for i in range(len(boxes)):
            box = boxes[i]
            predicted_class_name = labels[i]
            c_proba = int(np.round(scores[i],2)*100)
            # rectangle
            rect = patches.Rectangle((box[0], box[1]), abs(box[0]-box[2]), abs(box[1]-box[3]), linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # annotation
            text = f"{predicted_class_name}: {c_proba}%"
            if box[0] >= center_width:
                hoal = "right"
                offset = 0
                pta = box[0]+abs(box[0]-box[2])
            else:
                hoal = "left"
                offset = 0
                pta = box[0]
            ax.annotate(
                text,
                color="red",
                xy=(pta, box[1]), 
                xytext=(offset, 0),
                textcoords='offset points',
                ha=hoal, 
                va='bottom')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.show()


##############################################################################################################
# Part 3 functions
###### model training
### plot model training results
def plot_model_training(type_history, model_name, json_history_filepath=None, image_folderpath=None, history=None, output_folderpath=""):
    """
    Create loss and accuracy training plots (if they don't exist) and display them.

    Parameters
    ----------
    type_history: str
        The type of history: either "json" if you want use a .json file, "image" if you have already save them or "history" if you want to directly use the history object that comes from the training.
    model_name: str
        The name of the model.
    json_history_filepath: str, default=None
        If type_history="json", the filepath where the json file is saved.
    image_folderpath: str, default=None
        If type_history="image", the folderpath where the image file is saved.
    history: History, default=None
        If type_history="history", the History object that comes from the training.
    output_folderpath: str, default=""
        The folderpath where you want plots to be saved.

    Returns
    -------
        Nothing is returned.
    """
    ### if we want to plot from a .json file
    if type_history == "json":
        # get history file
        with open(json_history_filepath) as file:
            history = json.load(file)
        history["epoch"] = np.arange(1, len(history["loss"])+1).astype(int)
        # create the plots and save them
        plotly_model_training(history, model_name, save=True, output_folderpath=output_folderpath)
        # display them
        image_model_training(model_name, output_folderpath)

    elif type_history == "image":
        ## otherwise, if the plots images already exist
        image_model_training(model_name, image_folderpath)

    elif type_history == "history":
        ## otherwise, if the history object comes from training
        # history object
        new_history = history.history
        new_history["epoch"] = np.arange(1, len(new_history["loss"])+1).astype(int)
        # create the plots and save them
        plotly_model_training(new_history, model_name, save=True, output_folderpath=output_folderpath)
        # display them
        image_model_training(model_name, output_folderpath)

### display plots from saved images
def image_model_training(model_name, folderpath):
    """
    Display plots from saved images.

    Parameters
    ----------
    model_name: str
        The name of the model.
    folderpath: str
        The folderpath where the plots have been saved.
    
    Returns
    -------
        Nothing is returned.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    axs[0].imshow(Image.open(folderpath+model_name+"_loss_history.png"))
    axs[0].axis("off")
    axs[1].imshow(Image.open(folderpath+model_name+"_accuracy_history.png"))
    axs[1].axis("off")
    plt.show()

### create plotly visualizations of loss and accuracy
def plotly_model_training(history, model_name, save=False, output_folderpath=""):
    """
    Create and save plots of loss and accuracy evolution through epochs.

    Parameters
    ----------
    history: dict
        A dictionary containing training informations: "epochs", "loss", "val_loss", "accuracy", "val_accuracy".
    model_name: str
        The name of the model.
    save: bool, default=False
        If True, the plots will be saved.
    output_folderpath: str, default=""
        The folderpath where plots will be saved.

    Returns
    -------
        Nothing is returned.
    """
    width = 800 # plot width
    for metrics in ["loss", "accuracy"]:
        fig = go.Figure()
        # training
        fig.add_trace(go.Scatter(x=history["epoch"], y=history[metrics],
                            mode='lines+markers',
                            name='train '+metrics))
        # validation
        fig.add_trace(go.Scatter(x=history["epoch"], y=history["val_"+metrics],
                            mode='lines+markers',
                            name='val '+metrics))
        fig.update_layout(template="plotly_white", width=width, title="")
        fig.update_xaxes(title="epoch")
        fig.update_yaxes(title=metrics)
        # save
        if save:
            os.makedirs(output_folderpath, exist_ok=True)
            fig.write_image(output_folderpath+model_name+"_"+metrics+"_history.png")


###### confusion matrix
### plot confusion matrix
def plot_confusion_matrix_type(matrix_type, pred_type, class_names, conf_mat=None, output_folderpath=""):
    """
    Plot (and create and save) confusion matrix with plotly.

    Parameters
    ----------
    matrix_type: str
        The type of matrix to be displayed. Can be either "confusion", "accuracies" or "errors". 
    pred_type:
        A string corresponding to the prediction type, e.g.: "train", "val", "test", etc.
    class_names: list
        A list of strings corresponding to the names of classes.
    conf_mat: numpy.array or None
        A numpy matrix corresponding to the confusion matrix. If None, the image that should be in the output folderpath will be displayed.
    output_folderpath: str, default=""
        A folderpath where the plot will be saved.

    Returns
    -------
        Plotly figure or PIL Image.
    """
    if matrix_type == "confusion":
        if conf_mat is None:
            return Image.open(output_folderpath+pred_type+"_conf_mat.png").convert("RGB")
        else:
            os.makedirs(output_folderpath, exist_ok=True)
            fig = plot_confusion_matrix(conf_mat, class_names, save=True, filepath=output_folderpath+pred_type+"_conf_mat")
            return fig
    elif matrix_type == "accuracies":
        if conf_mat is None:
            return Image.open(output_folderpath+pred_type+"_acc_mat.png").convert("RGB")
        else:
            os.makedirs(output_folderpath, exist_ok=True)
            fig = plot_accuracies_matrix(conf_mat, class_names, save=True, filepath=output_folderpath+pred_type+"_acc_mat")
            return fig
    elif matrix_type == "errors":
        if conf_mat is None:
            return Image.open(output_folderpath+pred_type+"_err_mat.png").convert("RGB")
        else:
            os.makedirs(output_folderpath, exist_ok=True)
            fig = plot_error_matrix(conf_mat, class_names, save=True, filepath=output_folderpath+pred_type+"_err_mat")
            return fig
    
### confusion
def plot_confusion_matrix(conf_mat, class_names, save=False, filepath="conf_mat"):
    fig = px.imshow(
        conf_mat,
        labels=dict(x="Prediction", y="Actual", color="Number"),
        x=class_names,
        y=class_names,
        color_continuous_scale='Reds'
    )
    fig.update_xaxes(side="top")
    fig.update_layout(width=800, height=600)
    if save:
        fig.write_html(filepath+".html")
        fig.write_image(filepath+".png")
    return fig

### accuracies
def plot_accuracies_matrix(conf_mat, class_names, save=False, filepath="acc_mat"):
    accuracies = (conf_mat/conf_mat.sum(axis=1))*100
    fig = px.imshow(
        accuracies,
        labels=dict(x="Prediction", y="Actual", color="Percentage"),
        x=class_names,
        y=class_names,
        color_continuous_scale='Reds'
    )
    fig.update_xaxes(side="top")
    fig.update_layout(width=800, height=600)
    if save:
        fig.write_html(filepath+".html")
        fig.write_image(filepath+".png")
    return fig

### errors
def plot_error_matrix(conf_mat, class_names, save=False, filepath="err_mat"):
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mat.copy()/row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    fig = px.imshow(
        norm_conf_mx*100,
        labels=dict(x="Prediction", y="Actual", color="Percentage"),
        x=class_names,
        y=class_names,
        color_continuous_scale='Reds'
    )
    fig.update_xaxes(side="top")
    fig.update_layout(width=800, height=600)
    if save:
        fig.write_html(filepath+".html")
        fig.write_image(filepath+".png")
    return fig


###### predictions errors
### plot images side by side
def plot_class_confusion(n, nrows, ncols, class1, class2, folderpath, output="", set_type="train", random_state=123, gtsrb_exists=False):
    """
    Plot a number of class1 images on left columns and class2 on right columns.
    It can help in order to analyze confusions made by a classifier model.

    Parameters
    ----------
    n: int
        The total number of images to be displayed. Has to be equal to nrows times ncols.
    nrows: int
        The rows number.
    ncols: int
        The columns number. Has to be even.
    class1: int or str
        The name of a class.
    class2: int or str
        The name of another class.
    folderpath: str
        The folderpath where images have been saved.
    output: str, default=""
        The output filepath where you want the plot to be saved.
    set_type: str, default="train"
        A string corresponding to the set type, can be: "train", "val", "test".
    random_state: int, default=123
        An integer corresponding to the seed that will be use to ensure reproducibility .
    gtsrb_exists: bool , default=True
        If True, images will be taken from the dataset (has to be stored on "data/gtsrb/Train/"). Otherwise the image from the "imgs" will just be displayed.

    Returns
    -------
        Nothing is returned.
    """
    if gtsrb_exists:
        np.random.seed(random_state) # seed

        ## datasets folderpaths
        train_data_class1_dir = os.path.join(folderpath, class1)
        train_data_class2_dir = os.path.join(folderpath, class2)
        train_class1_filenames = os.listdir(train_data_class1_dir)
        train_class2_filenames = os.listdir(train_data_class2_dir)

        ## plot
        random_imgs = []
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
        for i in range(n):
            state = True
            cpt = 0
            row, col = i//ncols, i%ncols
            # handle images location
            if col < ncols//2:
                # right ones
                train_data_class_dir = train_data_class1_dir
                train_class_filenames = train_class1_filenames
            else:
                # left ones
                train_data_class_dir = train_data_class2_dir
                train_class_filenames = train_class2_filenames
            # get random images according to the ones already used
            while(state and (cpt < 10000)):
                random_idx = np.random.randint(len(train_class_filenames))
                random_img_file = train_class_filenames[random_idx]
                state = random_img_file in random_imgs
                cpt += 1
            # load the image
            img = Image.open(train_data_class_dir+"/"+random_img_file)
            #img = cv2.imread(os.path.join(train_data_class_dir, random_img_file))
            #img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            random_imgs.append(random_img_file)
            # plot
            axs[row, col].imshow(img)
            axs[row, col].axis("off")
        # filepath and save
        plt.tight_layout()
        if output == "":
            output = f"imgs/confusions/confusion_{class1}vs{class2}_{set_type}.png"
        plt.savefig(output)
        plt.show()

    else:
        return Image.open(f"imgs/confusions/confusion_{class1}vs{class2}_{set_type}.png").convert("RGB")

### plot wrong predictions given an actual and predicted classes
def plot_wrong_predictions(load_tf_ds, ds_location_data, plot_filepath, y, y_pred, actual, predicted, batch_size, gtsrb_exists=False):
    """
    Plot wrong predictions according to the chosen actual and predicted ones.

    Parameters
    ----------
    load_tf_ds: funciton
        A function defined with partial in order to reload dataset before getting predictions.
    ds_location_data: str
        The folderpath of the images set we want to predict.
    plot_filepath:
        The filepath to the file you want the plot to be saved.
    y: list
        The list of actual images labels.
    y_pred: list
        The list of predicted images labels.
    actual: int or str
        The name of an actual class.
    predicted: int or str
        The name of a predicted class.
    batch_size: int
        An integer corresponding to the batch size used for training.
    gtsrb_exists: bool, default=True
        If True, images will be taken from the dataset (has to be stored on "data/gtsrb/Train/"). Otherwise the image from the "imgs" will just be displayed.

    Returns
    -------
        Nothing is returned.
    """
    if gtsrb_exists:
        # reload the dataset
        loaded_ds = load_tf_ds(ds_location_data)
        # get prediction errors
        error_images = calculations.get_wrong_predictions(y, y_pred, loaded_ds, actual, predicted, batch_size)
        # plot images
        plot_images(error_images, plot_filepath)
    else:
        return Image.open(f"imgs/confusions/MN224-224-120da50_classes-{actual}-{predicted}.png").convert("RGB")

### plot a list of images
def plot_images(imgs, plot_filepath):
    """
    Plot images with a 10 columns layout.

    Parameters
    ----------
    imgs: list
        A list of images to plot.
    plot_filepath: str,
        A filepath to save the plot figure.

    Returns
    -------
        Nothing is returned.
    """
    n = len(imgs) # images number
    nrows = n//10 + 1*(n%10 != 0) # number of rows
    # plot
    fig, axs = plt.subplots(nrows=nrows, ncols=10, figsize=(10, nrows*3))
    for i in range(n):
        row, col = i//10, i%10
        img = imgs[i]
        axs[row, col].imshow(img/255)
        axs[row, col].axis("off")
    # empty spaces
    for i in range(n, nrows*10):
        axs[row, col].axis("off")
    # save and plot
    plt.savefig(plot_filepath)
    plt.plot()


##############################################################################################################
# Part 4 functions
### plot labeled images with object detection and classifier predictions
def labeled_image_with_object_detection_and_classifier_predictions(img, od_model, classifier, classes, od_threshold, classif_threshold, classif_img_dims, filepath):
    """
    Save a labeled image with predictions from object detection and classifier model.

    Parameters
    ----------
    img: numpy.array
        The numpy array you would get by reading an image this way:
        np.array(Image.open(filepath))
    od_model: detecto-like model
        An object detection model as the one created with the detecto package that can be used this way:
        labels, boxes, scores = od_model.predict(img)
    classifier: tensorflow model
        A classifier model as the one created in the notebooks.
    classes: list
        A list of classes you want keep to display predictions.
    od_threshold: float
        A probability threshold you want to apply to filter object detection model predictions. 
    classif_threshold: float
        A probability threshold you want to apply to filter classifier model predictions. 
    classif_img_dims: list or tuple
        A list or tuple of the classifier input height and width of images.
    filepath: str
        The filepath where you want the new image to be saved.

    Returns
    -------
        Nothing is returned.

    """
    global traffic_signs_names
    height, width = classif_img_dims

    ### get the image predictions
    labels, boxes, scores = od_model.predict(img)
    
    ### manage filters
    if len(labels) > 0:
        # convert tensor into numpy arrays
        boxes = boxes.numpy()
        scores = scores.numpy()
        # get classes filter indexes
        indexes = []
        for i in range(len(classes)):
            indexes.extend(np.where(np.array(labels) == classes[i])[0].tolist())
        indexes = np.array(indexes)
        if indexes.shape[0] > 0:
            labels, boxes, scores = np.array(labels)[indexes].tolist(), boxes[indexes], scores[indexes]
            # get predictions probability filter indexes
            indexes = np.where(scores >= od_threshold)[0]
            if indexes.shape[0] > 0:
                labels, boxes, scores = np.array(labels)[indexes].tolist(), boxes[indexes], scores[indexes]
            else:
                labels, boxes, scores = None, None, None
        else:
                labels, boxes, scores = None, None, None
    else:
        labels, boxes, scores = None, None, None

    # object detection and classifier predictions
    box_list, predicted_class_list, p_list = [], [], []
    if not(labels is None):
        n_predictions = len(labels)
        for j in range(n_predictions):
            c_box = boxes[j]
            # image prediction
            a, b, c, d = c_box.round().astype(int)
            predicted_img = img[b:d, a:c, :].copy()

            # class prediction
            n, m, _ = predicted_img.shape
            img = predicted_img.reshape(1, n, m, 3)
            resized_img = tf.keras.layers.Resizing(height, width)(img)
            pred = classifier.predict(resized_img)
            predicted_class, p = pred.argmax(), pred.max()
            
            box_list.append(c_box)
            predicted_class_list.append(predicted_class)
            p_list.append(p)

    ### plot
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig, ax = plt.subplots() 
    ax.imshow(img)
    center_width = img.shape[1]//2
    ## if there is no selected predictions, it'll just save the original image
    if boxes is None:
        pass
    else:
        # manage all the predictions
        for i in range(len(boxes)):
            box = boxes[i]
            predicted_class_name = predicted_class_list[i]
            p = p_list[i]

            # text
            predicted_class_name = traffic_signs_names[predicted_class_name] if p >= classif_threshold else "NaN"
            c_proba = int(np.round(p,2)*100)

            # rectangle
            rect = patches.Rectangle((box[0], box[1]), abs(box[0]-box[2]), abs(box[1]-box[3]), linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # annotation
            text = f"{predicted_class_name}: {c_proba}%"
            if box[0] >= center_width:
                hoal = "right"
                offset = 0
                pta = box[0]+abs(box[0]-box[2])
            else:
                hoal = "left"
                offset = 0
                pta = box[0]
            ax.annotate(
                text,
                color="red",
                xy=(pta, box[1]), 
                xytext=(offset, 0),
                textcoords='offset points',
                ha=hoal, 
                va='bottom')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.show()


