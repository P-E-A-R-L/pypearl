from enum import Enum

class VisualizationMethod(Enum):
    FEATURES  = 0 # a feature map <string, float>
    RGB_ARRAY = 1 # float 2d RGB image, where each pixel color is in the range of [0, 1.0]
    GRAY_SCAL = 2 # float 2d Gray image [0, 1.0]
    HEAT_MAP  = 3 # float 2d image [-inf, inf]
    BAR_CHART = 4 # a bar chart <string, float> where the string is the feature name and the float is the value