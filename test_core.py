import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import sklearn

print("=" * 50)
print("Core Libraries Test")
print("=" * 50)

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", plt.matplotlib.__version__)
print("Seaborn:", sns.__version__)
print("OpenCV:", cv2.__version__)
print("Pillow:", Image.__version__)
print("Scikit-Learn:", sklearn.__version__)

print("\n✅ All core libraries imported successfully!")