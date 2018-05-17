import tkinter as tk
from tkinter.filedialog import askopenfilename
import tkinter.ttk as ttk
from tkinter.constants import *
from PIL import Image, ImageTk
import glob
import os
import sys
import shutil
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import style
style.use('dark_background')

import Photometry 

class GUI(tk.Frame):
    
    @classmethod
    def main(cls):
        root = tk.Tk()
        app = cls(root)
        app.grid(sticky=NSEW)
        root.geometry('1350x850+50+50')
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)
        root.mainloop()

    def __init__(self, root):
        super().__init__(root)
        self.master = root
        self.pack(fill=BOTH, expand=True)

        self.tk_setPalette( background = 'gray5',
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
        self.ThresholdCompleted = False
        self.ChoiceToggle = False
        # Frame01 = tk.Frame(self)
        # Frame01.pack(fill=X)

        pack_opts = dict(fill=BOTH, padx=5, pady=2)
        grid_opts = dict(sticky=NSEW, padx=5, pady=2)



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
        Frame1 = tk.LabelFrame(self, text='Control')
        Frame1.pack(**pack_opts)
        ##################################################
        self.FolderName = tk.Entry(Frame1)
        self.FolderName.insert(0,
                f'{datetime.datetime.now().strftime("%Y%m%d_%H%M")}')
        self.FolderName.pack(side=LEFT, **pack_opts)

        self.runButton = tk.Button(Frame1, text='RUN', width=7, 
                height=1, command=self.run, bg='black',state=DISABLED)
        self.runButton.pack(side=LEFT, **pack_opts)

        self.openButton = tk.Button(Frame1, text='OPEN', width=7, 
                height=1, command=self.open, bg='black')
        self.openButton.pack(side=LEFT, **pack_opts)
       
        self.restartButton = tk.Button(Frame1, text='RESET', width=7, 
                height=1, command=self.restart, bg='black')
        self.restartButton.pack(side=LEFT, **pack_opts)

        self.QuitButton = tk.Button(Frame1, text="QUIT", 
                command=Frame1.quit, width=7)
        self.QuitButton.pack(side=LEFT, **pack_opts)

        self.CatalogValue = tk.Entry(Frame1)
        self.CatalogValue.insert(0,'0')
        self.CatalogValue.pack(side=RIGHT, **pack_opts)

        self.CatLab = tk.Label(Frame1, text='Reference Magnitude:')
        self.CatLab.pack(side=RIGHT, **pack_opts)

        ##################################################
        Frame2 = tk.Frame(self)
        Frame2.pack(**pack_opts)
        ##################################################

        ##################################################
        #Frame 2_1: Threshold Toggle Buttons
        ##################################################
        Frame2_1 = tk.Frame(Frame2, height=480)
        Frame2_1.pack(side=LEFT, **pack_opts)

        self.ThresholdVariable=tk.IntVar()
        self.ThresholdOn = tk.Radiobutton(Frame2_1, text="ON", 
                variable=self.ThresholdVariable, value=1, indicatoron=0, 
                command=self.ThresholdRadio, wraplength=1, width=4, height=17)
        self.ThresholdOff = tk.Radiobutton(Frame2_1, 
                text="OFF", variable=self.ThresholdVariable, value=0, height=17,
                indicatoron=0, command=self.ThresholdRadio, wraplength=1, width=4)
        self.ThresholdOn.pack(side=TOP, **pack_opts)
        self.ThresholdOff.pack(side=BOTTOM, **pack_opts)
        # self.ThresholdOn.grid(row=0, column=0, **grid_opts)
        # self.ThresholdOff.grid(row=1, column=0, **grid_opts)

        # Threshold Slider
        self.thresholdvar = tk.IntVar()
        self.thresholdvar.set(100)
        self.ThresholdSlider = tk.Scale(Frame2, from_=255, to=0, 
                command=self.Threshold,
                variable=self.thresholdvar, length=480, showvalue=False)
        self.ThresholdSlider.pack(side=LEFT, **pack_opts) 


        ##################################################
        #Frame 2_2: Video View and Slider
        ##################################################
        Frame2_2 = tk.Frame(Frame2)
        Frame2_2.pack(side=LEFT, **pack_opts)

        self.CamView = []
        self.canvas = tk.Canvas(Frame2_2, width = 720, height=480, bg='black')
        self.canvas.pack(side=TOP, **pack_opts)
        self.ImageCanvas= self.canvas.create_image(0,0, anchor=NW, 
                image=self.CamView)
        self.canvas.bind("<Button-1>",self.ReferenceCoordinates)
        self.canvas.bind("<Button-3>",self.ObjectCoordinates)

        self.slidervar = tk.IntVar()
        self.InitialSlider = tk.Scale(Frame2_2, from_=0, to=100, 
                orient=HORIZONTAL, label='Frame Slider', 
                command=self.VideoLength, variable=self.slidervar,length=720)
        self.InitialSlider.pack(side=TOP)


        ##################################################
        #Frame 2_3: Results View and Slider
        ##################################################
        Frame2_3 = tk.Frame(Frame2)
        Frame2_3.pack(side=LEFT, **pack_opts)

        self.display = tk.Label(Frame2_3, image="")
        self.display.pack(side=TOP, **pack_opts)

        self.var = tk.IntVar()
        self.FrameSlider = tk.Scale(Frame2_3, from_=1, to=2, 
                orient=HORIZONTAL, length = 480, command=self.PlotFrame, 
                variable=self.var, label='Plot Frame', state=tk.DISABLED )
        self.FrameSlider.pack(side=BOTTOM, **pack_opts)




        ##################################################
        Frame3 = tk.Frame(self)
        Frame3.pack(side=TOP, **pack_opts) 
        ##################################################

        ################################################## 
        #Frame 3_1: Clicked Locations
        ################################################## 
        Frame3_1 = tk.Frame(Frame3)
        Frame3_1.pack(side=LEFT, **pack_opts)

        self.Choice=tk.IntVar()
        self.ObjectLabel = tk.Button(Frame3_1, text="Object: ( X , Y )",
                command=self.ObjectChoice, height=5)
        self.ReferenceLabel = tk.Button(Frame3_1, height=5,
                text="Reference Star: ( X, Y )", command=self.ObjectChoice)
        self.ObjectLabel.pack(side=TOP, **pack_opts)
        self.ReferenceLabel.pack(side=TOP, **pack_opts)
        """ I realize by making these normal buttons I nuked some functionality
            here, so I'll debate on whether to reimplement that or not
        """


        self.fig = Figure(figsize=(11.5,2), dpi=100)
        self.plt = self.fig.add_subplot(111)
        self.plt.plot([1,2,3,4],[1,2,3,4], color='red')
        self.plt.set_xlabel('Time (s)')
        self.plt.set_ylabel('Magnitude')
        self.fig.set_tight_layout({'pad': 0.50})

        self.LightCurve = FigureCanvasTkAgg(self.fig, Frame3)
        self.LightCurve.draw()
        self.LightCurve.get_tk_widget().pack(side=LEFT, **pack_opts)
        # self.LightCurve._tkcanvas.grid(**grid_opts)

        # self.LightCurve = tk.Label(Frame3, image="")
        # self.LightCurve.pack(side=LEFT)


##################################################
# Functions
##################################################

    def PlotFrame(self,event):
        FrameNumber = self.var.get()
        if self.ChoiceToggle == False:
            PlotPath= f'{self.FolderPath}/ObjectPlot{FrameNumber:03}.png'
            self.PlotImage = Image.open(PlotPath)
            self.PlotImage = self.PlotImage.resize((480,480),Image.ANTIALIAS)
            self.PlotImage = ImageTk.PhotoImage(self.PlotImage)
            self.display.config(image=self.PlotImage)
        if self.ChoiceToggle == True:
            PlotPath= f'{self.FolderPath}/ReferencePlot{FrameNumber:03}.png'
            self.PlotImage = Image.open(PlotPath)
            self.PlotImage = self.PlotImage.resize((360,360),Image.ANTIALIAS)
            self.PlotImage = ImageTk.PhotoImage(self.PlotImage)
            self.display.config(image=self.PlotImage)

    def ReferenceCoordinates(self,event):
        self.ReferenceLabel.configure(text=f"Reference: ({event.x},{event.y})")
        self.REFERENCESTARLOC = (event.x,event.y)
        self.ReferenceLabel.configure(bg ='green',fg = 'black')
        self.ReferenceClick = True
        if self.ObjectClick == True and self.ThresholdCompleted == True:
            self.runButton.config(state='normal')
        if hasattr(self, 'refspot'):
            self.canvas.delete(self.refspot)
        self.refspot = self.canvas.create_oval(event.x-5,
                event.y-5,event.x+5,event.y+5, fill="", outline="green", width=2)


    def ObjectCoordinates(self,event):
        self.ObjectLabel.configure(text=f"Object: ({event.x},{event.y})")
        self.OBJECTLOC = (event.x,event.y)
        self.ObjectLabel.configure(bg ='firebrick4',fg ='black')
        self.ObjectClick = True
        if self.ReferenceClick == True and self.ThresholdCompleted == True:
            self.runButton.config(state='normal')
        if hasattr(self, 'objspot'):
            self.canvas.delete(self.objspot)
        self.objspot = self.canvas.create_oval(event.x-5,
                event.y-5,event.x+5,event.y+5, fill="", outline="firebrick4",
                width=2)
            
    def ObjectChoice(self):
        if self.Choice.get() == 0:
            self.ChoiceToggle = False
        if self.Choice.get() == 1:
            self.ChoiceToggle = True

    def ThresholdRadio(self):
        self.ThresholdSelection = self.ThresholdVariable.get()
        if self.ThresholdSelection == 0:
            self.ThresholdToggle = False
        if self.ThresholdSelection == 1:
            self.ThresholdToggle = True
        self.RefreshVideoImage()

    def VideoLength(self,event):
        self.FrameNo = self.slidervar.get()
        self.RefreshVideoImage()

    def RefreshVideoImage(self):
        if self.ThresholdToggle:
            self.ThresholdNumber = self.thresholdvar.get()
            ThresholdView = Photometry.Threshold(self.ImageStack[:,:,self.FrameNo], 
                    self.ThresholdNumber)
            self.CamView = Image.fromarray(ThresholdView)
            self.CamView = ImageTk.PhotoImage(self.CamView)
            self.canvas.itemconfig(self.ImageCanvas,image=self.CamView)
            self.ThresholdCompleted = True
        else:
            self.CamView = Image.fromarray(self.ImageStack[:,:,self.FrameNo])
            self.CamView = ImageTk.PhotoImage(self.CamView)
            self.canvas.itemconfig(self.ImageCanvas,image=self.CamView)

    def Threshold(self,event):
        self.ThresholdToggle = True
        self.ThresholdVariable.set(True)
        self.RefreshVideoImage()

    def open(self):
        self.VideoName = tk.filedialog.askopenfilename(initialdir = 'Data',
                filetypes=(("Video Files", "*.avi"),("All Files", "*.*")))
        self.VideoPath = '/'.join(self.VideoName.split('/')[:-1])
        self.FolderDirectory = (self.FolderName.get())
        self.FolderPath = f'{self.VideoPath}/{self.FolderDirectory}'

        self.ImageStack = Photometry.InitialRead(self.VideoName)
        VideoSize = self.ImageStack.shape[2]
        self.InitialSlider.config(to=VideoSize-1)
        self.RefreshVideoImage()
        # self.runButton.configure(bg='black',fg='white')
        # self.openButton.configure(bg='white', fg='black')
        self.openButton.configure(state=DISABLED)

    def run(self):
        # Only create folder in directory upon hitting run
        if os.path.exists(self.FolderPath):
            shutil.rmtree(self.FolderPath)
        os.makedirs(self.FolderPath)
        # Run script
        self.Catalog = self.CatalogValue.get()
        self.runButton.configure(text="Running")
        Photometry.main(self.VideoName,self.FolderPath,self.FrameNo,
                self.OBJECTLOC,self.REFERENCESTARLOC,
                self.ThresholdNumber,self.Catalog)
        # lightcurvepath= f'{self.FolderPath}/LightCurve.png'
        # self.OG = Image.open(lightcurvepath)
        # ResizedOG = self.OG.resize((660,140),Image.ANTIALIAS)
        # self.IMG = ImageTk.PhotoImage(ResizedOG)
        # self.LightCurve.config(image=self.IMG)
        # self.runButton.configure(bg = 'white', fg='black')
        MaxFrameNumber= len(glob.glob(f"{self.FolderPath}/ObjectPlot*.png"))
        self.FrameSlider.config(to=MaxFrameNumber, state=tk.NORMAL)
        self.restartButton.configure(bg ='firebrick4')
        self.runButton.configure(text="Run")

        FrameNumber = self.var.get()
        PlotPath= f'{self.FolderPath}/ObjectPlot{FrameNumber:03}.png'
        self.PlotImage = Image.open(PlotPath)
        self.PlotImage = self.PlotImage.resize((480,480),Image.ANTIALIAS)
        self.PlotImage = ImageTk.PhotoImage(self.PlotImage)
        self.display.config(image=self.PlotImage)

        self.updateLightCurve()

    def updateLightCurve(self, *args):
        df = pd.read_csv(f'{self.FolderPath}/Magnitudes.csv')
        self.plt.clear()
        self.fig.gca().invert_yaxis()
        self.plt.plot(df.Time, df.Object, color='red')
        self.plt.set_xlabel('Time (s)')
        self.plt.set_ylabel('Magnitude')
        self.fig.canvas.draw()

    def restart(self):
        python = sys.executable
        os.execl(python, python, * sys.argv)

if __name__ == '__main__':
    GUI.main()
