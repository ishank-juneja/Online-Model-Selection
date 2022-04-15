import numpy as np


class CachedData:
    """
    Wrapper around a npz archive/cache file being loaded into memory
    """
    def __init__(self, npz_pth: str):
        # List the fields required for our run cache
        self.required_atts = ['states', 'actions', 'obs']

        try:
            self.npz = np.load(npz_pth)
        except:
            raise FileNotFoundError("Passed file path {0} cannot be opened".format(npz_pth))

        # Check for fields in npz file
        self.check_npz_fields()

    def check_npz_fields(self):
        """
        Checks if all the expected fields are avail in the loaded npz archive
        :return:
        """
        # Pycharm complains since it thinks its np array not npz file
        avail_atts = self.npz.files
        for req_att in self.required_atts:
            if req_att not in avail_atts:
                raise ValueError("Incorrect npz cache missing entry {0}".format(req_att))

    def get_attribute(self, attr: str) -> np.ndarray:
        supported_atts = self.required_atts
        # Check if legal attribute
        if attr not in supported_atts:
            raise ValueError("Unsupported attribute {0} supplied".format(attr))
        else:
            return self.npz[attr]


if __name__ == '__main__':
    loader = CachedData("data/hand_made_tests/kendama-1.npz")

    attr_name = 'obs'
    attr = loader.get_attribute(attr_name)
    print(attr.shape)
