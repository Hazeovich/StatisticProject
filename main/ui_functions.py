from ui_graphics import *
import sys
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class UiFunctions(Ui_MainWindow):
    def __init__(self, window):
        self.setupUi(window)
        self.add_btn_func()
        self.sc = MplCanvas(MainWindow, width=5, height=4, dpi=100)
        self.toolbar = NavigationToolbar2QT(self.sc, MainWindow)
        self.show_plt()
        
    def show_plt(self):
        #self.sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])        
        self.plotLayout.addWidget(self.toolbar)
        self.plotLayout.addWidget(self.sc)
    
    def add_btn_func(self):
        self.btn_show_plt.clicked.connect(self.switch_settings_plt)
        
    def switch_settings_plt(self):
        self.sc.axes.plot([0, 1, 2, 3, 4], [10, 20, 30, 20, 10])
        self.sc.draw()
        print('asdad')
        
    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = UiFunctions(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())