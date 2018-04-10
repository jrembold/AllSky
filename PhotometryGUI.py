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

        self.tk_setPalette( background = 'black',
                            foreground = 'gray80',
                            activeBackground = 'gray30',
                            selectColor = 'firebrick4',
                            selectBackground = 'firebrick4' )

        self.createwidgets()

    def createwidgets(self):

        Frame1 = tk.Frame(self)
        Frame1.grid(row=0, column=1, sticky=NSEW, padx=5, pady=5)   

        CanvasWidth,CanvasHeight = 360,240
        self.Canvas = tk.Canvas(Frame1, 
                width=CanvasWidth, height=CanvasHeight,
                background='firebrick4')
        self.Canvas.grid(row=1,column=1, 
                rowspan=CanvasHeight, columnspan=CanvasWidth)

        BlankCanvasWidth,BlankCanvasHeight = 50,240
        self.BlankCanvas = tk.Canvas(Frame1, 
                width=BlankCanvasWidth, height=BlankCanvasHeight)
        self.BlankCanvas.grid(row=1,column=CanvasWidth+1, 
                rowspan=BlankCanvasHeight, columnspan=BlankCanvasWidth)
        
        ThresholdSliderLength,ThresholdSliderHeight = 360,15
        self.ThresholdSlider = tk.Scale(Frame1, from_=0, to=100, 
                orient=HORIZONTAL, label='Threshold', sliderlength = 60, 
                length=ThresholdSliderLength)
        self.ThresholdSlider.grid(row=CanvasHeight+1, column=1, 
                columnspan=ThresholdSliderLength, rowspan=ThresholdSliderHeight)
        
        InitialSliderLength,InitialSliderHeight = 360,15
        self.InitialSlider = tk.Scale(Frame1, from_=0, to=100, 
                orient=HORIZONTAL, label='Initial Frame', sliderlength = 60, 
                length=InitialSliderLength, command=self.VideoLength)
        self.InitialSlider.grid(row=CanvasHeight+ThresholdSliderHeight+1, column=1, 
                columnspan=InitialSliderLength, rowspan=InitialSliderHeight)

        RadioWidth,RadioHeight = 13,2
        v=tk.IntVar()
        self.Selection = tk.Radiobutton(Frame1, text="Object", 
                variable=v, value=0, indicatoron=0, 
                height=RadioHeight, width=RadioWidth)
        self.Selection2 = tk.Radiobutton(Frame1, text="Reference Star", 
                variable=v, value=1, indicatoron=0, 
                height=RadioHeight, width=RadioWidth)
        self.Selection.grid(row=0, column=650)
        self.Selection2.grid(row=0, column=700)

        FrameNumber = 1
        FolderDirectory = "Documents"
        # if Ran == True:
            # path= f'/home/luke/{FolderDirectory}/ObjectPlot{FrameNumber:03}.png'
            # print(path)
            # self.Original = Image.open(path)
            # Resized = self.Original.resize((340,340),Image.ANTIALIAS)
            # self.img = ImageTk.PhotoImage(Resized)
        self.display = tk.Label(Frame1, image="")
        self.display.grid(row=1,column=450,rowspan=340,columnspan=340)

        self.LightCurve = tk.Label(Frame1, image="")
        self.LightCurve.grid(row=350,column=411,rowspan=240,columnspan=340)

        FrameSliderLength,FrameSliderHeight = 290,15
        self.var = tk.IntVar()
        self.FrameSlider = tk.Scale(Frame1, from_=1, to=2, 
                orient=HORIZONTAL,  sliderlength = 10, 
                length = FrameSliderLength, command=self.FrameValue, 
                variable=self.var)
        self.FrameSlider.grid(row=600, column=650,
                columnspan=FrameSliderLength, rowspan=FrameSliderHeight) 


        self.QuitButton = tk.Button(Frame1, text="QUIT", 
                command=Frame1.quit, width=7)
        self.QuitButton.grid(row=0,column=3)

        self.FolderName = tk.Entry(Frame1)
        self.FolderName.insert(0,
                f'{datetime.datetime.now().strftime("%y-%m-%d-%H-%M")}')
        self.FolderName.grid(row=0,column=5)
        
        self.openButton = tk.Button(Frame1, text='OPEN', width=7, 
                height=1, command=self.open)
        self.openButton.grid(row=0, column=6)

##################################################
# Functions
##################################################

    def FrameValue(self,event):
        FrameNumber = self.var.get()
        path= f'/home/luke/{FolderDirectory}/ObjectPlot{FrameNumber:03}.png'
        self.Original = Image.open(path)
        Resized = self.Original.resize((340,340),Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(Resized)
        self.display.config(image=self.img)

    def VideoLength(self,event):
        global StartFrame
        StartFrame = self.var.get() 


    def open(self):
        global FolderDirectory
        global VideoName

        # Folder Creation
        FolderDirectory = (self.FolderName.get())
        FolderPath = f'/home/luke/{FolderDirectory}'
        if os.path.exists(FolderPath):
            shutil.rmtree(FolderPath)
        os.makedirs(FolderPath)
        VideoName = tk.filedialog.askopenfilename(title='Choose a file')

        ImageStack = Photometry.InitialRead(VideoName)
        VideoSize = ImageStack.shape[2]
        self.InitialSlider.config(to=VideoSize)

        #Run Script
        #Photometry.main(VideoName,FolderDirectory,StartFrame)

        #Graphs
        # lightcurvepath= f'/home/luke/{FolderDirectory}/LightCurve.png'
        # self.OG = Image.open(lightcurvepath)
        # ResizedOG = self.OG.resize((340,240),Image.ANTIALIAS)
        # self.IMG = ImageTk.PhotoImage(ResizedOG)
        # self.LightCurve.config(image=self.IMG) 
        # MaxFrameNumber= len(glob.glob("ObjectPlot*.png"))
        # self.FrameSlider.config(to=MaxFrameNumber)


if __name__ == '__main__':
    GUI.main()
