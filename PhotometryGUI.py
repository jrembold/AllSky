import tkinter as tk
from tkinter.filedialog import askopenfilename
import tkinter.ttk as ttk
from tkinter.constants import *
from PIL import Image, ImageTk
import glob
import Photometry 
import os
import shutil
import time
import datetime
import numpy as np
import cv2

Ran = False

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

        self.QuitButton = tk.Button(Frame1, text="QUIT", 
                command=Frame1.quit, width=7)
        self.QuitButton.pack(side=LEFT)

        self.runButton = tk.Button(Frame1, text='RUN', width=7, 
                height=1, command=self.run)
        self.runButton.pack(side=LEFT)

        self.openButton = tk.Button(Frame1, text='OPEN', width=7, 
                height=1, command=self.open)
        self.openButton.pack(side=LEFT)

        self.FolderName = tk.Entry(Frame1)
        self.FolderName.insert(0,
                f'{datetime.datetime.now().strftime("%y-%m-%d-%H-%M")}')
        self.FolderName.pack(side=LEFT)
       
        self.progress= ttk.Progressbar(Frame1, orient=HORIZONTAL, length=200, mode='determinate',maximum=100)
        self.progress.pack(side=LEFT)

        ##################################################
        Frame2 = tk.Frame(self)
        Frame2.pack(fill=X) 
        ##################################################

        self.CamView = []
        self.canvas = tk.Canvas(Frame2, width = 720, height=480, bg='black')
        self.canvas.pack(side=LEFT)
        self.ImageCanvas= self.canvas.create_image(0,0, anchor=NW, 
                image=self.CamView)
        self.canvas.bind("<Button-1>",self.ReferenceCoordinates)
        self.canvas.bind("<Button-3>",self.ObjectCoordinates)

        ##################################################
        Frame3 = tk.Frame(self)
        Frame3.pack(fill=X)
        ##################################################


        self.slidervar = tk.IntVar()
        self.InitialSlider = tk.Scale(Frame3, from_=0, to=100, 
                orient=HORIZONTAL, label='Initial Frame', 
                command=self.VideoLength, variable=self.slidervar,length=360)
        self.InitialSlider.pack(side=LEFT)
        
        self.thresholdvar = tk.IntVar()
        self.ThresholdSlider = tk.Scale(Frame3, from_=0, to=255, 
                orient=HORIZONTAL, label='Threshold',
                command=self.Threshold,variable=self.thresholdvar,length=360)
        self.ThresholdSlider.pack(side=LEFT)
        
        ##################################################
        Frame4 = tk.Frame(self)
        Frame4.pack(fill=X)
        ##################################################
       
        self.LightCurve = tk.Label(Frame4, image="")
        self.LightCurve.pack(side=LEFT)

        self.display = tk.Label(Frame4, image="")
        self.display.pack(side=LEFT)
        
        ##################################################
        Frame5=tk.Frame(self)
        Frame5.pack(fill=X)
        ##################################################

        self.var = tk.IntVar()
        self.FrameSlider = tk.Scale(Frame5, from_=1, to=2, 
                orient=HORIZONTAL, length = 360, command=self.PlotFrame, 
                variable=self.var)
        self.FrameSlider.pack(side=LEFT)

        # RadioWidth,RadioHeight = 13,2
        # v=tk.IntVar()
        # self.Selection = tk.Radiobutton(Frame1, text="Object", 
                # variable=v, value=0, indicatoron=0, 
                # height=RadioHeight, width=RadioWidth)
        # self.Selection2 = tk.Radiobutton(Frame1, text="Reference Star", 
                # variable=v, value=1, indicatoron=0, 
                # height=RadioHeight, width=RadioWidth)
        # self.Selection.grid(row=0, column=4)
        # self.Selection2.grid(row=0, column=5)

##################################################
# Functions
##################################################

    def PlotFrame(self,event):
        FrameNumber = self.var.get()
        PlotPath= f'Data/{self.FolderDirectory}/ObjectPlot{FrameNumber:03}.png'
        self.PlotImage = Image.open(PlotPath)
        self.PlotImage = self.PlotImage.resize((360,360),Image.ANTIALIAS)
        self.PlotImage = ImageTk.PhotoImage(self.PlotImage)
        self.display.config(image=self.PlotImage)

    def ReferenceCoordinates(self,event):
        print(event.x,event.y)
        self.REFERENCESTARLOC = (event.x,event.y)

    def ObjectCoordinates(self,event):
        print(event.x,event.y)
        self.OBJECTLOC = (event.x,event.y)

    def VideoLength(self,event):
        self.FrameNo = self.slidervar.get()
        self.CamView = Image.fromarray(self.ImageStack[:,:,self.FrameNo])
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
        # Folder Creation
        self.FolderDirectory = (self.FolderName.get())
        FolderPath = f'Data/{self.FolderDirectory}'
        if os.path.exists(FolderPath):
            shutil.rmtree(FolderPath)
        os.makedirs(FolderPath)
        self.VideoName = tk.filedialog.askopenfilename(title='Choose a file')
        self.ImageStack = Photometry.InitialRead(self.VideoName)
        VideoSize = self.ImageStack.shape[2]
        self.InitialSlider.config(to=VideoSize-1)
        self.progress.config(maximum=VideoSize)
        self.CamView = Image.fromarray(self.ImageStack[:,:,0])
        CamView = self.CamView.resize((640,480),Image.ANTIALIAS)
        self.CamView = ImageTk.PhotoImage(CamView)
        self.canvas.itemconfig(self.ImageCanvas,image=self.CamView)

    def run(self):
        Photometry.main(self.VideoName,self.FolderDirectory,self.FrameNo,
                self.OBJECTLOC,self.REFERENCESTARLOC,self.ThresholdNumber)
        lightcurvepath= f'Data/{self.FolderDirectory}/LightCurve.png'
        self.OG = Image.open(lightcurvepath)
        ResizedOG = self.OG.resize((360,360),Image.ANTIALIAS)
        self.IMG = ImageTk.PhotoImage(ResizedOG)
        self.LightCurve.config(image=self.IMG)
        print('1111')
        MaxFrameNumber= len(glob.glob(f"Data/{self.FolderDirectory}/ObjectPlot*.png"))
        self.FrameSlider.config(to=MaxFrameNumber)

if __name__ == '__main__':
    GUI.main()
