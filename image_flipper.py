from PIL import Image
import os
from config import *


def flip_image(image_path, flip_direction):
    """
    Flip an image horizontally or vertically and save the result as a JPEG file.

    Args:
        image_path (str): Path to the image file to flip.
        flip_direction (str): Either 'horizontal' or 'vertical' to specify the direction of the flip.

    Raises:
        ValueError: If flip_direction is not 'horizontal' or 'vertical'.

    Returns:
        str: Path to the saved flipped image file.
    """
    # Open the image file
    image = Image.open(image_path)

    # Flip the image
    if flip_direction == "horizontal":
        flipped_image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
    elif flip_direction == "vertical":
        flipped_image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
    else:
        raise ValueError("flip_direction must be 'horizontal' or 'vertical'")

    # Save the flipped image as a JPEG file
    flipped_image_path = image_path[:-4] + f"_flipped_{flip_direction}.jpg"
    flipped_image.save(flipped_image_path)

    return


for Class in classes:
    directory = os.path.join(data_root, data_folder)
    # Loop over every file in the directory
    for filename in os.listdir(directory):
        # Check if the file is an image
        if filename.endswith(".jpg"):
            # Flip the image vertically and horizontally
            for flip_direction in ["vertical", "horizontal"]:
                flipped_image_path = flip_image(os.path.join(directory, filename), flip_direction)
