

class ImageDataDir:
    def __init__(self, folder_pth: str, extension: str):
        self.folder = folder_pth
        self.extension = extension


class AugmentationDataDirs:
    """
    A config file mapping the path names of all image augmentations related datasets
    """
    def __init__(self):
        # Filled shape files generated using blender
        self.filled_shapes_og = ImageDataDir("data/filled_shapes/", '*.bmp')
        # Filled shape models made to look like simulator shapes in Blender
        # by mimicking lighting and texture of Mujoco env shapes
        self.filled_shapes = ImageDataDir("data/filled_shapes/expanded", '*.npy')
        # Outline shapes generated using outline_shapes_mjco.py script
        self.outline_shapes = ImageDataDir("data/outline_shapes/", '*.npy')
        # A folder from the full ImageNet dataset
        self.image_net = ImageDataDir("data/ILSVRC/Data/DET/test", '*.JPEG')
        # A texture from the kaggle textures paint by numbers challenge
        # https://www.kaggle.com/c/painter-by-numbers
        self.textures = ImageDataDir("data/textures", '*.jpg')
        # Simple model augmentations
        self.ball = ImageDataDir("data/ball_aug", "*.npy")
        self.cartpole = ImageDataDir("data/cartpole_aug", "*.npy")
        self.dcartpole = ImageDataDir("data/dcartpole_aug", "*.npy")
        self.dubins = ImageDataDir("data/dubins_aug", "*.npy")
