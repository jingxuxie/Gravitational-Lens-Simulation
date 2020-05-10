# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:19:42 2020

@author: HP
"""

from PyQt5.QtWidgets import QWidget, QSlider, QVBoxLayout, QHBoxLayout, \
    QLabel, QApplication, QGridLayout, QPushButton, QCheckBox, QAction, \
    QFileDialog, QMainWindow, QDesktopWidget, QToolButton, QComboBox,\
    QMessageBox, QProgressBar, QSplashScreen, QLineEdit, QShortcut, QMenu,\
    QColorDialog, qApp
from PyQt5.QtCore import Qt, QThread, QTimer, QObject, pyqtSignal, QBasicTimer, \
    QEvent
from PyQt5.QtGui import QPixmap, QImage, QIcon
import sys
import os
import time
import cv2
import numpy as np
from auxiliary_func import get_folder_from_file
from curve import black_hole_simulation


class Label(QLabel):
    new_img = pyqtSignal(str)
    def __init__(self, title, parent):
        super().__init__(title, parent)
        self.setAcceptDrops(True)
        self.support_format = ['jpg', 'png', 'bmp']
        
    def dragEnterEvent(self, e):
        m = e.mimeData()
        if m.hasUrls():
            if m.urls()[0].toLocalFile()[-3:] in self.support_format:
                e.accept()
            else:
                e.ignore()
        else:
            e.ignore()
    
    def dropEvent(self, e):
        m = e.mimeData()
        if m.hasUrls():
            self.img = cv2.imread(m.urls()[0].toLocalFile())
            
#            self.setPixmap(QPixmap(m.urls()[0].toLocalFile()))
            self.new_img.emit('new')
#            print(m.urls()[0].toLocalFile())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        
        self.current_dir = os.path.abspath(__file__).replace('\\','/')
        self.current_dir = get_folder_from_file(self.current_dir)
        self.current_dir += 'support_files/'
        
        self.blanck = cv2.imread(self.current_dir + 'blanck.jpg')
        self.blanck_simu = cv2.imread(self.current_dir + 'blanck_simu.jpg')
        self.img = self.blanck
        print(self.current_dir)
#        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        self.time_start = 0
        self.time_usage = 50
        
        self.canvas_blank = np.zeros((512,512),dtype = np.int8)
        
        openAct = QAction(QIcon(self.current_dir + 'open.png'), '&Open', self)
        openAct.setShortcut('Ctrl+F')
        openAct.triggered.connect(self.open_file)
        
        
        exitAct = QAction(QIcon(self.current_dir + 'quit.png'), '&Quit', self)        
        exitAct.setShortcut('Ctrl+Q')
        exitAct.triggered.connect(qApp.quit)
        
        help_contact = QAction(QIcon(self.current_dir + 'email.png'), 'Contact', self)
        help_contact.triggered.connect(self.contact)
        
        help_about = QAction('About', self)
        help_about.triggered.connect(self.about)
        
        self.menubar = self.menuBar()
        
        FileMenu = self.menubar.addMenu('&File')
        FileMenu.addAction(openAct)
        FileMenu.addAction(exitAct)
        
        HelpMenu = self.menubar.addMenu('&Help')
        HelpMenu.addAction(help_contact)
        HelpMenu.addAction(help_about)
        
        self.open_button = QToolButton()
        self.open_button.setIcon(QIcon(self.current_dir + 'galery.png'))
        self.open_button.setToolTip('Open File Ctr+F')
        self.open_button.clicked.connect(self.open_file)
        self.open_button.setShortcut('Ctr+F')
        
        self.run_button = QToolButton()
        self.run_button.setIcon(QIcon(self.current_dir + 'run.png'))
        self.run_button.setToolTip('Run F5')
#        self.run_button.clicked.connect(self.run_cut)
        self.run_button.setShortcut('F5')

        
        self.toolbar1 = self.addToolBar('Read')
        self.toolbar1.addWidget(self.open_button)
        
        
        self.pixmap = QPixmap()
        self.lbl_main = Label('',self)
        self.lbl_main.new_img.connect(self.refresh_img)
#        self.lbl_main.setAlignment(Qt.AlignTop)
        
#        self.lbl_main.setAlignment(Qt.AlignCenter)
        self.lbl_main.setPixmap(self.pixmap)
        
        self.img_qi = QImage(self.blanck[:], self.blanck.shape[1], self.blanck.shape[0],\
                          self.blanck.shape[1] * 3, QImage.Format_RGB888)
        self.pixmap = QPixmap(self.img_qi)
        self.lbl_main.setPixmap(self.pixmap)
        
        
        img_qi = QImage(self.blanck_simu[:], self.blanck_simu.shape[1], self.blanck_simu.shape[0],\
                          self.blanck_simu.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap(img_qi)
        self.lbl_simu = QLabel(self)
        self.lbl_simu.setPixmap(pixmap)
        
        
        self.lbl_L1 = QLabel('L_1', self)
        self.text_L1 = QLineEdit('1', self)
        self.lbl_L1_ly = QLabel('light year(s)', self)
        
        self.lbl_L2 = QLabel('L_2', self)
        self.text_L2 = QLineEdit('1', self)
        self.lbl_L2_ly = QLabel('light year(s)', self)
        
        
        self.lbl_mass = QLabel('M', self)
        self.text_mass = QLineEdit('1e11', self)
        self.lbl_msun = QLabel('M_sun', self)
        
        self.lbl_height = QLabel('height', self)
        self.text_height = QLineEdit('1', self)
        self.lbl_height_ly = QLabel('light year(s)', self)
        
        
        self.start_button = QPushButton('Start Simulation', self)
        self.start_button.clicked.connect(self.start_simulation)
        
        self.pbar = QProgressBar(self)
        self.step = 0
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.step_pbar)
        
#        self.stop_button = QPushButton('Stop', self)
#        self.stop_button.clicked.connect(self.stop_simulation)
        
        
        self.panel = QGridLayout()
        self.panel.addWidget(self.lbl_L1, 0, 0)
        self.panel.addWidget(self.text_L1, 0, 1)
        self.panel.addWidget(self.lbl_L1_ly, 0, 2)
        self.panel.addWidget(self.lbl_L2, 1, 0)
        self.panel.addWidget(self.text_L2, 1, 1)
        self.panel.addWidget(self.lbl_L2_ly, 1, 2)
        self.panel.addWidget(self.lbl_mass, 2, 0)
        self.panel.addWidget(self.text_mass, 2, 1)
        self.panel.addWidget(self.lbl_msun, 2, 2)
        self.panel.addWidget(self.lbl_height, 3, 0)
        self.panel.addWidget(self.text_height, 3, 1)
        self.panel.addWidget(self.lbl_height_ly, 3, 2)
        self.panel.addWidget(self.start_button, 4, 1)
        self.panel.addWidget(self.pbar, 5, 1)
        
        self.vbox = QVBoxLayout()
        self.vbox.addStretch(0)
        self.vbox.addLayout(self.panel)
        self.vbox.addStretch(0)
        
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.lbl_main)
        self.hbox.addWidget(self.lbl_simu)
        self.hbox.addLayout(self.vbox, Qt.AlignRight)
        
        
        self.central_widget = QWidget()
        self.central_widget.setMouseTracking(True)
       
        self.layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)
        self.layout.addLayout(self.hbox)
        
        desktop = QDesktopWidget()
        self.screen_width = desktop.screenGeometry().width()
        self.screen_height = desktop.screenGeometry().height()
        self.img_width = int((self.screen_width - 500) / 2)
        self.img_height = int(self.screen_height - 100)
        print(self.screen_height, self.screen_width)
        
        self.setWindowIcon(QIcon(self.current_dir+'lensing.png'))
        self.setWindowTitle('Gravitational Lens Simulation')
        self.show()
        
    
    def contact(self):
        QMessageBox.information(self, 'contact','Please contact jingxuxie@berkeley.edu.'+\
                                ' Thanks!')
    
    def about(self):
        QMessageBox.information(self, 'About', 'Gravitational Lens Simulation. '+ \
                                'Proudly designed and created by Jingxu Xie(谢京旭).\n \n'
                                'Copyright © 2020 Jingxu Xie. All Rights Reserved.')
    
    def open_file(self):
        self.fname = QFileDialog.getOpenFileName(self, 'Open file',\
                                                 self.current_dir + 'examples',\
                                                 "Image files(*.jpg *.png *.bmp)")
        if self.fname[0]:
            self.lbl_main.img = cv2.imread(self.fname[0])
            self.refresh_img('new')
    
    def refresh_img(self, s):
        if s == 'new':
            self.img = self.lbl_main.img
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            if self.img.shape[1] > self.img_width:
                rate = self.img_width / self.img.shape[1]
                self.img = cv2.resize(self.img, (self.img_width, int(self.img.shape[0] * rate)))
                
            if self.img.shape[0] > self.img_height:
                rate = self.img_height / self.img.shape[0]
                self.img = cv2.resize(self.img, (int(self.img.shape[1] * rate), self.img_height))
            
            self.img_qi = QImage(self.img[:], self.img.shape[1], self.img.shape[0],\
                          self.img.shape[1] * 3, QImage.Format_RGB888)
            self.pixmap = QPixmap(self.img_qi)
            self.lbl_main.setPixmap(self.pixmap)
            print(self.img.shape)
    
    def start_simulation(self):
        self.time_start = time.time()
        L_1 = float(self.text_L1.text()) * 1e16
        L_2 = float(self.text_L2.text()) * 1e16
        M = float(self.text_mass.text()) * 1477
        D = float(self.text_height.text()) * 1e16
        self.simulation = Simulation_Thread(self.img, L_1, L_2, M, D)
        self.simulation.finished.connect(self.recv_new_simulation)
        self.simulation.start()
        self.step = 0
        self.pbar.setValue(0)
        self.progress_timer.start(int(1000*self.time_usage/400))
     
    def step_pbar(self):
        if self.step <= 98:
            self.step += 1
            self.pbar.setValue(self.step)
        else:
            self.progress_timer.stop()
        
    def stop_simulation(self):
        self.simulation.terminate()
#        self.simulation.exit()
        self.simulation.quit()
#        self.simulation.threadactive = False
#        self.simulation.wait()
        
    def recv_new_simulation(self, s):
        if s == 'finished':
            self.img_simu = self.simulation.img_new
            self.time_usage = time.time() - self.time_start
            self.progress_timer.stop()
            self.pbar.setValue(100)
            if (self.img_simu.shape == np.array([512, 512, 3])).all():
                if (self.img_simu == np.zeros((512, 512, 3))).all():
                    self.img_simu = cv2.imread(self.current_dir + 'error.jpg')
            
            if self.img_simu.shape[1] > self.img_width:
                rate = self.img_width / self.img_simu.shape[1]
                self.img_simu = cv2.resize(self.img_simu, (self.img_width, int(self.img_simu.shape[0] * rate)))
                
            if self.img_simu.shape[0] > self.img_height:
                rate = self.img_height / self.img_simu.shape[0]
                self.img_simu = cv2.resize(self.img_simu, (int(self.img_simu.shape[1] * rate), self.img_height))
            
            self.img_simu = cv2.cvtColor(self.img_simu, cv2.COLOR_BGR2RGB)
            img_qi = QImage(self.img_simu[:], self.img_simu.shape[1], self.img_simu.shape[0],\
                              self.img_simu.shape[1] * 3, QImage.Format_RGB888)
            pixmap = QPixmap(img_qi)
            self.lbl_simu.setPixmap(pixmap)
            self.simulation.terminate()
        

class Simulation_Thread(QThread):
    finished = pyqtSignal(str)
    def __init__(self, img, L_1 = 1e5, L_2 = 1e5, M = 1477, D = 1e5):
        super().__init__()
        self.img = img
        self.L_1 = L_1
        self.L_2 = L_2
        self.M = M
        self.D = D
        
    def run(self):
        self.img_new = black_hole_simulation(self.img, self.L_1, self.L_2, self.M, self.D)
        self.finished.emit('finished')
        






if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    splash_path = os.path.abspath(__file__).replace('\\','/')
    splash_path = get_folder_from_file(splash_path)
    splash = QSplashScreen(QPixmap(splash_path + 'support_files/lensing.png'))
    print(splash_path + 'support_files/lensing.png')
    splash.show()
    splash.showMessage('Loading……')
    
    window = MainWindow()
    
    splash.close()
    sys.exit(app.exec_())
    
    
    