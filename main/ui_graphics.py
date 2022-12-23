# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setMinimumSize(QtCore.QSize(800, 600))
        MainWindow.setMaximumSize(QtCore.QSize(800, 600))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(6, 10, 791, 581))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.plotLayout = QtWidgets.QGridLayout()
        self.plotLayout.setObjectName("plotLayout")
        self.verticalLayout.addLayout(self.plotLayout)
        self.buttonLayout = QtWidgets.QHBoxLayout()
        self.buttonLayout.setContentsMargins(40, -1, 40, -1)
        self.buttonLayout.setSpacing(40)
        self.buttonLayout.setObjectName("buttonLayout")
        self.btn_show_plt = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btn_show_plt.setObjectName("btn_show_plt")
        self.buttonLayout.addWidget(self.btn_show_plt)
        self.btn_show_zz = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btn_show_zz.setObjectName("btn_show_zz")
        self.buttonLayout.addWidget(self.btn_show_zz)
        self.btn_show_hist = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btn_show_hist.setObjectName("btn_show_hist")
        self.buttonLayout.addWidget(self.btn_show_hist)
        self.verticalLayout.addLayout(self.buttonLayout)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Statistic project"))
        self.btn_show_plt.setText(_translate("MainWindow", "show plot"))
        self.btn_show_zz.setText(_translate("MainWindow", "show zig-zag"))
        self.btn_show_hist.setText(_translate("MainWindow", "show histogram"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
