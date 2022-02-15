import os
import glob
import shutil
import re


# Class to translate between absolute and relative paths while reading/writing
# programmatically generated results in python scripts
# User specifies a human readable loc_name for every location of interest and a relative path for the location initially
# Later while using user only deals with convenient loc_name while absolute and relative paths are handled internally
class ResultDirManager:
    # Optionally initialize file_paths manager object as
    def __init__(self):
        # Dict to hold folder types and the (relative) path locations of those folder types
        # Example {'imgs': 'results/images', 'arrays': 'data/arrays_folder'}
        self.mypaths = {}
        # Dict to hold folder types and the absolute path locations of those folder types
        # Example {'imgs': 'home/ishank/Desktop/project/results/images',
        # 'arrays': 'home/ishank/Desktop/project/data/arrays_folder'}
        self.myabspaths = {}
        # Get absolute current working directory
        self.mycwd = os.getcwd()
        # My locations/keys
        self.mylocs = []

    # Takes a relative path name and location and creates a dir if there is none
    # NOTE: Relative to os.getcwd which may be differnt from the dir in which this .py file itself is present
    def add_location(self, loc_name: str, loc_rel_pth: str, make_dir_if_none: bool = True):
        self.mylocs.append(loc_name)
        # Assemble absolute path for loc_rel_pth
        abs_loc_pth = os.path.join(self.mycwd, loc_rel_pth)
        # Check if loc_dir actually exists and is a dir, if not make the folder
        if not os.path.isdir(abs_loc_pth) and make_dir_if_none:
            os.makedirs(abs_loc_pth)
        # Add to dicts
        self.mypaths[loc_name] = loc_rel_pth
        # Add to abs dicts
        self.myabspaths[loc_name] = os.path.join(self.mycwd, loc_rel_pth)
        return self.myabspaths[loc_name]

    def loc_exists(self, loc_name: str):
        if loc_name in self.mylocs:
            return True
        else:
            raise KeyError("Unknown location name {0} for dict with keys {1}".format(loc_name, self.mylocs))

    # Get the relative path for a certain location name
    def get_rel_path(self, loc_name: str):
        if self.loc_exists(loc_name):
            return self.mypaths[loc_name]

    # Get the abs path for a certain location name
    def get_abs_path(self, loc_name: str):
        if self.loc_exists(loc_name):
            return self.myabspaths[loc_name]

    # Get the absolute path for a file name at a certain location
    def get_file_path(self, loc_name: str, file_name: str, check_exists: bool = False):
        if self.loc_exists(loc_name):
            file_abs_path = os.path.join(self.myabspaths[loc_name], file_name)
            if check_exists:
                if not os.path.exists(file_abs_path):
                    raise ValueError('No file at {0}'.format(file_abs_path))
            return file_abs_path

    # Courtesy: https://stackoverflow.com/questions/17984809/how-do-i-create-an-incrementing-filename-in-python
    # Finds the next free path in a sequentially named list of files via exponential search
    def next_path(self, loc_name: str, prefix: str, postfix: str):
        if self.loc_exists(loc_name):
            pass
        # Append default or custom postfix to prefix
        # Example prefix = 'trajplt' and postfix = '-%s.png' to assemble file names of the form trajplt-i.png, i \in N
        name_w_pattern = prefix + postfix
        full_pattern_pth = os.path.join(self.get_abs_path(loc_name), name_w_pattern)
        i = 1
        # First do an exponential search for what files already exist
        while os.path.exists(full_pattern_pth % i):
            i = i * 2
        # Result lies somewhere in the interval (i/2..i]
        # We call this interval (a..b] and narrow it down until a + 1 = b
        a, b = (i // 2, i)
        while a + 1 < b:
            c = (a + b) // 2  # interval midpoint
            a, b = (c, b) if os.path.exists(full_pattern_pth % c) else (a, c)
        return full_pattern_pth % b

    # Scrapes a location (folder) to find files exactly matching a certain prefix
    # In the file names everything except for the prefix should be an integer
    def scrape_loc_for_prefix(self, loc_name: str, prefix: str):
        if self.loc_exists(loc_name):
            paths = glob.glob(os.path.join(self.get_abs_path(loc_name), prefix))
            # Sort the files in the order of their integer post-fixes
            sorted_paths = self.natural_sort(paths)
            return sorted_paths
        else:
            raise KeyError("Invalid location {0} specified, available locations are {1}".format(loc_name, self.mylocs))

    def list_dir_objects(self, abs_path_to_dir: str, pattern: str, return_sorted: bool = True):
        if pattern is not None:
            objects = glob.glob(os.path.join(abs_path_to_dir, pattern))
        else:
            objects = os.listdir(abs_path_to_dir)
        object_list = []
        for object_name in objects:
            object_list.append(os.path.join(abs_path_to_dir, object_name))
        if return_sorted:
            return self.natural_sort(object_list)
        else:
            return object_list

    def natural_sort(self, l):
        return sorted(l, key=self._alphanum_key)

    def _tryint(self, s):
        try:
            return int(s)
        except:
            return s

    def _alphanum_key(self, s):
        """
        Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        :param s:
        :return:
        """
        return [self._tryint(c) for c in re.split('([0-9]+)', s)]

    # make a new dir by removing existing dir if necessary
    def make_fresh_dir(self, loc_name: str, dir_name: str, over_write: bool = True):
        # Assemble absolute path for desired directory
        abs_dir_path = os.path.join(self.get_abs_path(loc_name), dir_name)
        if os.path.isdir(abs_dir_path):
            if over_write:
                shutil.rmtree(abs_dir_path)
            else:
                return True
        os.makedirs(abs_dir_path)
        return abs_dir_path

    # Make a new folder with name based on a dictionary of paramaters
    # For instance {size: 5, length 2} will make a folder: size_5_length_2
    def make_dir_from_dict(self, loc_name: str, mydict: dict, prefix: str = None, suffix: str = None):
        if self.loc_exists(loc_name):
            dir_name = ""
            for key, value in mydict.items():
                dir_name += key + '_' + str(value) + '_'
            # Remove trailing '_'
            dir_name = dir_name[:-1]
            if prefix is not None:
                dir_name = prefix + dir_name
            if suffix is not None:
                dir_name = dir_name + suffix
            # Assemble path, skip the end point `_`
            dir_path = os.path.join(self.get_abs_path(loc_name), dir_name)
            self.make_fresh_dir(loc_name, dir_path)
            return dir_path

    # function to strip a file path of the file extension to get a dir path corresponding to a folder name. Example
    # strip /home/ishank/myarray.npy to get the path to a new dir /home/ishank/myarray for storing things about my array
    def strip_path_of_extension(self, file_path: str):
        # Split path by . and extract out first element
        return file_path.split('.')[0]

    def get_name_from_path(self, file_path: str):
        # Split path by / and extract out last element
        return file_path.split('/')[-1]
