from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def preprocess_frame(frame):
    """
    Preprocess a raw Atari RGB frame into a normalized 1D feature vector.

    Steps:
    1. Crop irrelevant parts of the image (score and borders).
    2. Convert RGB image to grayscale by averaging color channels.
    3. Resize image to 80x80 pixels.
    4. Convert pixel values to float32.
    5. Normalize pixel values to range [0, 1].
    6. Flatten the 2D image into a 1D vector.

    Args:
        frame (np.ndarray): Raw RGB frame with shape (210, 160, 3).

    Returns:
        np.ndarray: Preprocessed frame as a 1D float32 array of shape (6400,).
    """

    # 1. Crop
    frame = frame[35:195]

    # 2. Conver to grayscale
    frame = np.mean(frame, axis=2)

    # 3. Resize to 80x80
    image = Image.fromarray(frame)
    image = image.resize((80, 80))

    frame = np.array(image)

    # 4. Convert to float32
    frame = frame.astype(np.float32)

    # 5. Normalization
    frame /= 255.0

    # 6. Flatten
    frame = frame.flatten()

    return frame


def preprocess_with_difference(frame, previous_frame):
    """
    Compute the difference between the current and previous preprocessed frames.

    This function helps capture motion information by highlighting changes
    between consecutive frames, which is critical for Atari games like Pong.

    Args:
        frame (np.ndarray): Current raw RGB frame.
        previous_frame (np.ndarray or None): Previous preprocessed frame.

    Returns:
        tuple:
            - np.ndarray: Frame difference (motion representation).
            - np.ndarray: Current preprocessed frame.
    """

    current_frame = preprocess_frame(frame)

    if previous_frame is None:
        frame_difference = np.zeros_like(current_frame)
    else:
        frame_difference = current_frame - previous_frame

    return frame_difference, current_frame



def save_image(frame, name, shape=None, cmap=None):
    """
    Save a frame as an image file.

    Optionally reshapes the input vector into an image before saving.

    Args:
        frame (np.ndarray): Input frame (flattened or image array).
        name (str): Output filename without extension.
        shape (tuple, optional): Shape to reshape the frame into (e.g., (80, 80)).
        cmap (str, optional): Colormap used for saving grayscale images.
    """
    img = frame
    
    if shape is not None:
        img = img.reshape(shape)

    plt.imsave(
       f"{name}.png",
       img,
       cmap=cmap
    )

def show_image(frame, title, shape=None, cmap=None):
    """
    Display a frame using matplotlib.

    Optionally reshapes the input vector into an image before visualization.

    Args:
        frame (np.ndarray): Input frame (flattened or image array).
        title (str): Title of the displayed plot.
        shape (tuple, optional): Shape to reshape the frame into (e.g., (80, 80)).
        cmap (str, optional): Colormap for grayscale visualization.
    """
    img = frame
    if shape is not None:
        img = img.reshape(shape)
    
    plt.imshow(
    img,
    cmap=cmap
    )

    plt.title(title)
    plt.show()
