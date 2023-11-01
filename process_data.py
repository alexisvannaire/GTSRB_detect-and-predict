


## libraries
import os
import pathlib
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import random


##############################################################################################################
# Part 1 functions
### random crop images
def random_crop(input_folder, output_folder, width, height):
    """
    Crops randomly 3 times the images that are in the input folder and save them in the output folder.

    Parameters
    ----------
    input_folder: str
        A folderpath corresponding to images that will be cropped.
    output_folder: str
        A folderpath where cropped images will be saved.
    width: int
        An integer corresponding to the wanted output image width.
    height: int
        An integer corresponding to the wanted output image height.

    Returns
    -------
        Nothing is returned.
    """
    # check if output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # list all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # loop on each image
    for img_file in image_files:
        # load image
        img_path = os.path.join(input_folder, img_file)
        with Image.open(img_path) as img:
            if img.width < width or img.height < height:
                print(f"Skipping {img_file} as its dimensions are smaller than the desired crop size.")
                continue
            
            # crop images 3 times
            for j in range(3):
                # generate random left top corner coordinates for cropping
                left = random.randint(0, img.width - width)
                top = random.randint(0, img.height - height)

                # perform cropping
                cropped_img = img.crop((left, top, left + width, top + height))
                
                # convert RGBA to RGB
                if cropped_img.mode == 'RGBA':
                    rgb_image = Image.new('RGB', cropped_img.size, (255, 255, 255))  # white background
                    rgb_image.paste(cropped_img, mask=cropped_img.split()[3])        # 3 is the alpha channel
                    cropped_img = rgb_image
                
                # save cropped image to the output folder
                img_new_filename = img_file.split("_")[0]
                cropped_img.save(os.path.join(output_folder, f"{img_new_filename}-{j}.jpg"), 'JPEG')

### add black rectangles into images
def add_black_rectangles(imgs_folderpath, output_folderpath, topleft, topright, bottomright, bottomleft, verbose=True):
    """
    Add black rectangles into images.

    Parameters
    ----------
    imgs_folderpath: str
        A folderpath to the input images.
    output_folderpath: str
        A folderpath where images will be saved.
    topleft: tuple or list
        A tuple or a list of two integers corresponding to the bottom-right coordinates of the top-left rectangle.
    topright: tuple or list
        A tuple or a list of two integers corresponding to the bottom left coordinates of the top-right rectangle.
    bottomright: tuple or list
        A tuple or a list of two integers corresponding to the top left coordinates of the bottom-right rectangle.
    bottomleft: tuple or list
        A tuple or a list of two integers corresponding to the top right coordinates of the bottom-left rectangle.
    verbose: bool, default=True
        Whether you want the running state to be displayed.

    Returns
    -------
        Nothing is returned.


    Note: All images should have the same heights and widths.
    """
    imgs_filenames = os.listdir(imgs_folderpath)
    n = len(imgs_filenames)
    ## loop on files
    for i in range(n):
        filename = imgs_filenames[i]
        # read img
        img = np.array(Image.open(imgs_folderpath+filename))
        # add black rectangles
        new_img = remove_google_maps_widgets(img, topleft, topright, bottomright, bottomleft)
        # save
        Image.fromarray(new_img, "RGB").save(output_folderpath+filename)
        if verbose:
            print(f"\r{i+1}/{n} : {filename}    ", end="")

### remove google maps widgets
def remove_google_maps_widgets(new_image, topleft, topright, bottomright, bottomleft):
    """
    Add black rectangles (pixels values to 0) into images.

    Parameters
    ----------
    new_image: numpy.array
        The numpy array corresponding to the image that will be modified.
    topleft: tuple or list
        A tuple or a list of two integers corresponding to the bottom-right coordinates of the top-left rectangle.
    topright: tuple or list
        A tuple or a list of two integers corresponding to the bottom left coordinates of the top-right rectangle.
    bottomright: tuple or list
        A tuple or a list of two integers corresponding to the top left coordinates of the bottom-right rectangle.
    bottomleft: tuple or list
        A tuple or a list of two integers corresponding to the top right coordinates of the bottom-left rectangle.

    Returns
    -------
        numpy.array
        The numpy array of the modified image.
    """
    # top left
    x, y = topleft
    new_image[:x, :y, :] = [0,0,0]
    # top right
    x, y = topright
    new_image[:x, y:, :] = [0,0,0]
    # bottom right
    x, y = bottomright
    new_image[x:, y:, :] = [0,0,0]
    # bottom left
    x, y = bottomleft
    new_image[x:, :y, :] = [0,0,0]

    return new_image


##############################################################################################################
# Part 2 functions
### update xml paths
def update_xml_paths():
    """
    Update google maps labeled images XML files.
    """
    # folderpath to XML files
    xml_folderpath = os.path.join(os.getcwd(), "data", "google_maps_test", "labeled_imgs")
    # XML file names
    xml_filenames = list(filter(lambda x: pathlib.Path(x).suffix == '.xml', os.listdir(xml_folderpath)))
    for xml_filename in xml_filenames:
        # XML file path
        xml_filepath = os.path.join(xml_folderpath, xml_filename)
        # Parse the XML file
        tree = ET.parse(xml_filepath)
        root = tree.getroot()
        # Replace the path of the XML file
        for path_element in root.findall(".//path"):
            path_element.text = xml_filepath
        # Save the changes back to the XML file
        tree.write(xml_filepath)


