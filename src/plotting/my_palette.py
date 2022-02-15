import matplotlib


# Class for stuff that every palette should have
class BasePalette:
    def __init__(self):
        pass

    def lineTypes(self):
        # TODO: Hard code in all line types -- , .- etc. into easy to use format
        return

    # Add patterns and fills to the default colors when they act as fills
    def addPatterns(self):
        # TODO: add this functionality based on plotting scripts from vidStab project
        return


# Class for high contrast colors distinguishable in a B&W print out
class HighContrast(BasePalette):
    def __init__(self):
        super().__init__()
        self.myPaletteList = []
        self.myPaletteDict = {}
        # Hard coded in (r, g, b, alpha) tuples which is a way matpltlib accepts colors
        # Add Blue
        self.myPaletteList.append((0., 0., 1., 1.))
        self.myPaletteDict['blue'] = (0., 0., 1., 1.)
        # Add Green
        self.myPaletteList.append((0.0, 0.5019607843137255, 0.0, 1.0))
        self.myPaletteDict['green'] = (0.0, 0.5019607843137255, 0.0, 1.0)
        # Add Red
        self.myPaletteList.append((1.0, 0.0, 0.0, 1.0))
        self.myPaletteDict['red'] = (1.0, 0.0, 0.0, 1.0)
        # Add Orange
        self.myPaletteList.append((1.0, 0.5490196078431373, 0.0, 1.0))
        self.myPaletteDict['orange'] = (1.0, 0.5490196078431373, 0.0, 1.0)

    def __getitem__(self, key):
        # If key is a string use dict, if int use list
        if isinstance(key, str):
            return self.myPaletteDict[key]
        elif isinstance(key, int):
            return self.myPaletteList[key]
        else:
            raise TypeError("Key must be either of type string (str) or int, found type {0} instead".format(type(key)))


# TODO: Currently a duplicate of HighContrast, use a tool like CMasher https://cmasher.readthedocs.io/user/introduction.html#example-use to
#  get these colors, also the anmswers here https://stackoverflow.com/questions/25408393/getting-individual-colors-from-a-color-map-in-matplotlib
#  and the qualitative color maps here https://matplotlib.org/stable/gallery/color/colormap_reference.html
#  if the above don't work out can use these 20 hard coded color values https://sashamaps.net/docs/resources/20-colors/
# 10 Regular colors for colored plotting use cases
class RegularColors(BasePalette):
    def __init__(self):
        super().__init__()
        self.myPaletteList = []
        self.myPaletteDict = {}
        # Hard coded in (r, g, b, alpha) tuples which is a way matpltlib accepts colors
        # Add Blue
        self.myPaletteList.append((0., 0., 1., 1.))
        self.myPaletteDict['blue'] = (0., 0., 1., 1.)
        # Add Green
        self.myPaletteList.append((0.0, 0.5019607843137255, 0.0, 1.0))
        self.myPaletteDict['green'] = (0.0, 0.5019607843137255, 0.0, 1.0)
        # Add Red
        self.myPaletteList.append((1.0, 0.0, 0.0, 1.0))
        self.myPaletteDict['red'] = (1.0, 0.0, 0.0, 1.0)
        # Add Orange
        self.myPaletteList.append((1.0, 0.5490196078431373, 0.0, 1.0))
        self.myPaletteDict['orange'] = (1.0, 0.5490196078431373, 0.0, 1.0)

    def __getitem__(self, key):
        # If key is a string use dict, if int use list
        if isinstance(key, str):
            return self.myPaletteDict[key]
        elif isinstance(key, int):
            return self.myPaletteList[key]
        else:
            raise TypeError("Key must be either of type string (str) or int, found type {0} instead".format(type(key)))

# For later, creating a color palatte algorithmically from an image https://towardsdatascience.com/algorithmic-color-palettes-a110d6448b5d
