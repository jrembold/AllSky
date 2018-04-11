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
        Frame1.grid(row=0, column=0, sticky=NSEW, padx=5, pady=5)   

        # self.ImageDisplay = tk.Label(Frame1, image="")
        # self.ImageDisplay.grid(row=5,column=0)

        self.CamView = []
        self.canvas = tk.Canvas(Frame1, width = 720, height=480, bg='red')
        self.canvas.grid(row=1,column=1)
        self.ImageCanvas= self.canvas.create_image(0,0, anchor=NW, image=self.CamView)
        self.canvas.bind("<Button-1>",self.ReferenceCoordinates)
        self.canvas.bind("<Button-3>",self.ObjectCoordinates)

        ThresholdSliderLength,ThresholdSliderHeight = 360,15
        self.thresholdvar = tk.IntVar()
        self.ThresholdSlider = tk.Scale(Frame1, from_=0, to=255, 
                orient=HORIZONTAL, label='Threshold', sliderlength = 60, 
                length=ThresholdSliderLength, command=self.Threshold,variable=self.thresholdvar)
        self.ThresholdSlider.grid(row=2, column=1)
        
        InitialSliderLength,InitialSliderHeight = 360,15
        self.slidervar = tk.IntVar()
        self.InitialSlider = tk.Scale(Frame1, from_=0, to=100, 
                orient=HORIZONTAL, label='Initial Frame', sliderlength = 60, 
                length=InitialSliderLength, command=self.VideoLength, 
                variable=self.slidervar)
        self.InitialSlider.grid(row=3, column=1)

        RadioWidth,RadioHeight = 13,2
        v=tk.IntVar()
        self.Selection = tk.Radiobutton(Frame1, text="Object", 
                variable=v, value=0, indicatoron=0, 
                height=RadioHeight, width=RadioWidth)
        self.Selection2 = tk.Radiobutton(Frame1, text="Reference Star", 
                variable=v, value=1, indicatoron=0, 
                height=RadioHeight, width=RadioWidth)
        self.Selection.grid(row=0, column=4)
        self.Selection2.grid(row=0, column=5)


        self.display = tk.Label(Frame1, image="")
        self.display.grid(row=1,column=4)

        self.LightCurve = tk.Label(Frame1, image="")
        self.LightCurve.grid(row=1,column=2)

        FrameSliderLength,FrameSliderHeight = 290,15
        self.var = tk.IntVar()
        self.FrameSlider = tk.Scale(Frame1, from_=1, to=2, 
                orient=HORIZONTAL,  sliderlength = 10, 
                length = FrameSliderLength, command=self.PlotFrame, 
                variable=self.var)
        self.FrameSlider.grid(row=2, column=4,
                columnspan=FrameSliderLength, rowspan=FrameSliderHeight) 


        self.QuitButton = tk.Button(Frame1, text="QUIT", 
                command=Frame1.quit, width=7)
        self.QuitButton.grid(row=0,column=1)

        self.FolderName = tk.Entry(Frame1)
        self.FolderName.insert(0,
                f'{datetime.datetime.now().strftime("%y-%m-%d-%H-%M")}')
        self.FolderName.grid(row=0,column=3)
        
        self.openButton = tk.Button(Frame1, text='OPEN', width=7, 
                height=1, command=self.open)
        self.openButton.grid(row=0, column=2)

        self.runButton = tk.Button(Frame1, text='RUN', width=7, 
                height=1, command=self.run)
        self.runButton.grid(row=0, column=3)

##################################################
# Functions
##################################################

    def PlotFrame(self,event):
        FrameNumber = self.var.get()
        PlotPath= f'/home/luke/{self.FolderDirectory}/ObjectPlot{FrameNumber:03}.png'
        self.PlotImage = Image.open(PlotPath)
        self.PlotImage = self.Original.resize((340,340),Image.ANTIALIAS)
        self.PlotImage = ImageTk.PhotoImage(PlotImage)
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
        ThresholdView = Photometry.Threshold(self.ImageStack[:,:,self.FrameNo], self.ThresholdNumber)
        self.CamView = Image.fromarray(ThresholdView)
        self.CamView = ImageTk.PhotoImage(self.CamView)
        self.canvas.itemconfig(self.ImageCanvas,image=self.CamView)


    def open(self):
        # Folder Creation
        self.FolderDirectory = (self.FolderName.get())
        FolderPath = f'/home/luke/{self.FolderDirectory}'
        if os.path.exists(FolderPath):
            shutil.rmtree(FolderPath)
        os.makedirs(FolderPath)
        self.VideoName = tk.filedialog.askopenfilename(title='Choose a file')

        self.ImageStack = Photometry.InitialRead(self.VideoName)
        VideoSize = self.ImageStack.shape[2]
        self.InitialSlider.config(to=VideoSize-1)
        self.CamView = Image.fromarray(self.ImageStack[:,:,0])
        CamView = self.CamView.resize((640,480),Image.ANTIALIAS)
        self.CamView = ImageTk.PhotoImage(CamView)
        self.canvas.itemconfig(self.ImageCanvas,image=self.CamView)


    def run(self):
        Photometry.main(self.VideoName,self.FolderDirectory,self.FrameNo,
                self.OBJECTLOC,self.REFERENCESTARLOC,self.ThresholdNumber)
        lightcurvepath= f'/home/luke/{self.FolderDirectory}/LightCurve.png'
        self.OG = Image.open(lightcurvepath)
        ResizedOG = self.OG.resize((340,240),Image.ANTIALIAS)
        self.IMG = ImageTk.PhotoImage(ResizedOG)
        self.LightCurve.config(image=self.IMG) 
        MaxFrameNumber= len(glob.glob("ObjectPlot*.png"))
        self.FrameSlider.config(to=MaxFrameNumber)


if __name__ == '__main__':
    GUI.main()
