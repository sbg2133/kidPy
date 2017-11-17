import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pygetdata as gd

# choose the newest dirfile in the folder
from glob import glob
files = glob('data/array/*')
files.sort()
filename = files[-1]

# open it for reading
df = gd.dirfile(filename,gd.RDONLY|gd.UNENCODED)

# make a numpy array to hold the array value
Nrows = int(df.field_list()[-2][1:4])
Ncols = int(df.field_list()[-2][5:])

def data_function():
    for i in xrange(Nrows):
        for j in xrange(Ncols):
            fieldname = 'r'+str(i).zfill(3)+'c'+str(j).zfill(3)
            data_function.data[i,j] = df.getdata(fieldname,gd.FLOAT64,first_frame=df.nframes-1,num_frames=1)
    return

data_function.data = np.zeros((Nrows,Ncols))

pg.mkQApp()

win = pg.GraphicsLayoutWidget()
win.setWindowTitle('pyqtgraph example: Image Analysis')

# A plot area (ViewBox + axes) for displaying the image
p1 = win.addPlot()

# Item for displaying image data
img = pg.ImageItem()
p1.addItem(img)

# Contrast/color control
hist = pg.HistogramLUTItem()
hist.setImageItem(img)
win.addItem(hist)

# Another plot area for displaying ROI data
win.resize(1000, 1000)
win.show()

img.setImage(data_function.data)
hist.setLevels(data_function.data.min(), data_function.data.max())

# set position and scale of image
img.scale(0.2, 0.2)
#img.translate(-50, 0)

# zoom to fit imageo
p1.autoRange()

# function to update the image
def update():
    data_function()
    img.setImage(data_function.data)
    hist.setLevels(data_function.data.min(), data_function.data.max())
    return

# timer to update the image
timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

QtGui.QApplication.instance().exec_()
