import tkinter as tk
from tkinter.filedialog import askopenfilename
import tkinter.ttk as ttk
from tkinter.constants import *
from PIL import Image, ImageTk
import glob
import Photometry 
import os
import sys
import shutil
import time
import datetime
import numpy as np
import cv2


class GUI(tk.Frame):
    
    @classmethod
    def main(cls):
        root = tk.Tk()
        app = cls(root)
        app.grid(sticky=NSEW)
        root.geometry('800x500+50+50')
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)
        root.mainloop()

    def __init__(self, root):
        super().__init__(root)
        self.master = root
        self.pack(fill=BOTH, expand=True)

        self.tk_setPalette( background = 'black',
                            foreground = 'gray80',
                            activeBackground = 'gray30',
                            selectColor = 'firebrick4',
                            selectBackground = 'firebrick4' )

        self.createwidgets()

    def createwidgets(self):
        self.Plotted = False
        self.ObjectClick = False
        self.ReferenceClick = False
        self.ThresholdToggle = False
        self.FrameNo = 1
        # Frame01 = tk.Frame(self)
        # Frame01.pack(fill=X)



        # mess = r"""
  # ______ _          _           _ _                        _           _     
 # |  ____(_)        | |         | | |     /\               | |         (_)    
 # | |__   _ _ __ ___| |__   __ _| | |    /  \   _ __   __ _| |_   _ ___ _ ___ 
 # |  __| | | '__/ _ \ '_ \ / _` | | |   / /\ \ | '_ \ / _` | | | | / __| / __|
 # | |    | | | |  __/ |_) | (_| | | |  / ____ \| | | | (_| | | |_| \__ \ \__ \
 # |_|    |_|_|  \___|_.__/ \__,_|_|_| /_/    \_\_| |_|\__,_|_|\__, |___/_|___/
                                                              # __/ |          
                                                             # |___/           
                                                             # """
        # self.Line=tk.Label(Frame01, justify=tk.LEFT, text=mess)
        # self.Line.pack()

        ################################################## 
        Frame1 = tk.Frame(self)
        Frame1.pack(fill=X)
        ##################################################
        self.FolderName = tk.Entry(Frame1)
        self.FolderName.insert(0,
                f'{datetime.datetime.now().strftime("%Y%m%d_%H%M")}')
        self.FolderName.pack(side=LEFT, fill=Y)

        self.runButton = tk.Button(Frame1, text='RUN', width=7, 
                height=1, command=self.run, bg='black',state=DISABLED)
        self.runButton.pack(side=LEFT)

        self.openButton = tk.Button(Frame1, text='OPEN', width=7, 
                height=1, command=self.open, bg='black')
        self.openButton.pack(side=LEFT)
       
        # self.ProgressLabel = tk.Button(Frame1, text="Not Started")
        # self.ProgressLabel.pack(side=LEFT)

        # self.ObjectLabel = tk.Button(Frame1, text="Object:( X , Y )")
        # self.ObjectLabel.pack(side=LEFT)

        # self.ReferenceLabel = tk.Button(Frame1, text="Reference:( X , Y )")
        # self.ReferenceLabel.pack(side=LEFT)

        self.restartButton = tk.Button(Frame1, text='RESET', width=7, 
                height=1, command=self.restart, bg='black')
        self.restartButton.pack(side=LEFT)

        self.QuitButton = tk.Button(Frame1, text="QUIT", 
                command=Frame1.quit, width=7)
        self.QuitButton.pack(side=LEFT)

        ##################################################
        Frame2 = tk.Frame(self)
        Frame2.pack(fill=X)
        ##################################################
        self.NameButton = tk.Button(Frame2, text="POSITION", width=7)
        self.NameButton.pack(side=LEFT)

        v=tk.IntVar()
        self.ObjectLabel = tk.Radiobutton(Frame2, text="Object: ( X , Y )", 
                variable=v, value=0, indicatoron=0)
        self.ReferenceLabel = tk.Radiobutton(Frame2, 
                text="Reference Star: ( X, Y )", variable=v, value=1, 
                indicatoron=0)
        self.ObjectLabel.pack(side=LEFT,fill=Y)
        self.ReferenceLabel.pack(side=LEFT,fill=Y)

        self.CatalogValue = tk.Entry(Frame2)
        self.CatalogValue.insert(0,'0')
        self.CatalogValue.pack(side=LEFT, fill=Y)


        ##################################################
        Frame2_1 = tk.Frame(self)
        Frame2_1.pack(fill=X) 
        ##################################################

        self.ThresholdVariable=tk.IntVar()
        self.ThresholdOn = tk.Radiobutton(Frame2_1, text="Threshold On", 
                variable=self.ThresholdVariable, value=1, indicatoron=0, command=self.ThresholdRadio)
        self.ThresholdOff = tk.Radiobutton(Frame2_1, 
                text="Threshold Off", variable=self.ThresholdVariable, value=0, 
                indicatoron=0, command=self.ThresholdRadio)
        self.ThresholdOn.pack(side=LEFT,fill=Y)
        self.ThresholdOff.pack(side=LEFT,fill=Y)


        ##################################################
        Frame3 = tk.Frame(self)
        Frame3.pack(fill=X) 
        ##################################################

        self.CamView = []
        self.canvas = tk.Canvas(Frame3, width = 720, height=480, bg='black')
        self.canvas.pack(side=LEFT)
        self.ImageCanvas= self.canvas.create_image(0,0, anchor=NW, 
                image=self.CamView)
        self.canvas.bind("<Button-1>",self.ReferenceCoordinates)
        self.canvas.bind("<Button-3>",self.ObjectCoordinates)

        self.LightCurve = tk.Label(Frame3, image="")
        self.LightCurve.pack(side=LEFT)

        self.display = tk.Label(Frame3, image="")
        self.display.pack(side=LEFT)
        ##################################################
        Frame4 = tk.Frame(self)
        Frame4.pack(fill=X)
        ##################################################


        self.slidervar = tk.IntVar()
        self.InitialSlider = tk.Scale(Frame4, from_=0, to=100, 
                orient=HORIZONTAL, label='Initial Frame', 
                command=self.VideoLength, variable=self.slidervar,length=360)
        self.InitialSlider.pack(side=LEFT)
        
        self.thresholdvar = tk.IntVar()
        self.ThresholdSlider = tk.Scale(Frame4, from_=0, to=255, 
                orient=HORIZONTAL, label='Threshold',
                command=self.Threshold,variable=self.thresholdvar,length=360)
        self.ThresholdSlider.pack(side=LEFT) 

        self.var = tk.IntVar()
        self.FrameSlider = tk.Scale(Frame4, from_=1, to=2, 
                orient=HORIZONTAL, length = 360, command=self.PlotFrame, 
                variable=self.var, label='Plot Frame' )
        self.FrameSlider.pack(side=LEFT)


##################################################
# Functions
##################################################

    def PlotFrame(self,event):
        FrameNumber = self.var.get()
        PlotPath= f'{self.FolderPath}/ObjectPlot{FrameNumber:03}.png'
        self.PlotImage = Image.open(PlotPath)
        self.PlotImage = self.PlotImage.resize((480,480),Image.ANTIALIAS)
        self.PlotImage = ImageTk.PhotoImage(self.PlotImage)
        self.display.config(image=self.PlotImage)

    def ReferenceCoordinates(self,event):
        self.ReferenceLabel.configure(text=f"Reference: ({event.x},{event.y})")
        self.REFERENCESTARLOC = (event.x,event.y)
        self.ReferenceLabel.configure(bg ='white',fg = 'black')
        self.ReferenceClick = True
        if self.ObjectClick == True:
            self.runButton.config(state='normal')

    def ObjectCoordinates(self,event):
        self.ObjectLabel.configure(text=f"Object: ({event.x},{event.y})")
        self.OBJECTLOC = (event.x,event.y)
        self.ObjectLabel.configure(bg ='white',fg ='black')
        self.ObjectClick = True
        if self.ReferenceClick == True:
            self.runButton.config(state='normal')
            
    def ThresholdRadio(self):
        self.ThresholdSelection = self.ThresholdVariable.get()
        if self.ThresholdSelection == 0:
            self.ThresholdToggle = False
        if self.ThresholdSelection == 1:
            self.ThresholdToggle = True
    def VideoLength(self,event):
        self.FrameNo = self.slidervar.get()
        if self.ThresholdToggle == False:
            self.CamView = Image.fromarray(self.ImageStack[:,:,self.FrameNo])
            self.CamView = ImageTk.PhotoImage(self.CamView)
            self.canvas.itemconfig(self.ImageCanvas,image=self.CamView)
        if self.ThresholdToggle == True:
            self.ThresholdNumber = self.thresholdvar.get()
            ThresholdView = Photometry.Threshold(self.ImageStack[:,:,self.FrameNo], 
                    self.ThresholdNumber)
            self.CamView = Image.fromarray(ThresholdView)
            self.CamView = ImageTk.PhotoImage(self.CamView)
            self.canvas.itemconfig(self.ImageCanvas,image=self.CamView)

    
    def Threshold(self,event):
        self.ThresholdNumber = self.thresholdvar.get()
        ThresholdView = Photometry.Threshold(self.ImageStack[:,:,self.FrameNo], 
                self.ThresholdNumber)
        self.CamView = Image.fromarray(ThresholdView)
        self.CamView = ImageTk.PhotoImage(self.CamView)
        self.canvas.itemconfig(self.ImageCanvas,image=self.CamView)

    def open(self):
        self.VideoName = tk.filedialog.askopenfilename(initialdir = 'Data')
        self.VideoPath = '/'.join(self.VideoName.split('/')[:-1])
        self.FolderDirectory = (self.FolderName.get())
        self.FolderPath = f'{self.VideoPath}/{self.FolderDirectory}'
        if os.path.exists(self.FolderPath):
            shutil.rmtree(self.FolderPath)
        os.makedirs(self.FolderPath)

        self.ImageStack = Photometry.InitialRead(self.VideoName)
        VideoSize = self.ImageStack.shape[2]
        self.InitialSlider.config(to=VideoSize-1)
        self.CamView = Image.fromarray(self.ImageStack[:,:,0])
        # CamView = self.CamView.resize((640,480),Image.ANTIALIAS)
        self.CamView = ImageTk.PhotoImage(self.CamView)
        self.canvas.itemconfig(self.ImageCanvas,image=self.CamView)
        self.openButton.configure(bg='white', fg='black')
        # self.runButton.configure(bg='black',fg='white')

    def run(self):
        self.Catalog = self.CatalogValue.get()
        print(self.Catalog)
        # self.runButton.configure(text="Running")
        Photometry.main(self.VideoName,self.FolderPath,self.FrameNo,
                self.OBJECTLOC,self.REFERENCESTARLOC,self.ThresholdNumber,self.Catalog)
        lightcurvepath= f'{self.FolderPath}/LightCurve.png'
        self.OG = Image.open(lightcurvepath)
        ResizedOG = self.OG.resize((480,360),Image.ANTIALIAS)
        self.IMG = ImageTk.PhotoImage(ResizedOG)
        self.LightCurve.config(image=self.IMG)
        self.runButton.configure(bg = 'white', fg='black')
        MaxFrameNumber= len(glob.glob(f"{self.FolderPath}/ObjectPlot*.png"))
        self.FrameSlider.config(to=MaxFrameNumber)
        self.restartButton.configure(bg ='firebrick4')

        FrameNumber = self.var.get()
        PlotPath= f'{self.FolderPath}/ObjectPlot{FrameNumber:03}.png'
        self.PlotImage = Image.open(PlotPath)
        self.PlotImage = self.PlotImage.resize((480,480),Image.ANTIALIAS)
        self.PlotImage = ImageTk.PhotoImage(self.PlotImage)
        self.display.config(image=self.PlotImage)

    def restart(self):
        python = sys.executable
        os.execl(python, python, * sys.argv)

if __name__ == '__main__':
    GUI.main()
