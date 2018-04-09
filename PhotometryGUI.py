import tkinter as tk
from tkinter.filedialog import askopenfilename
import tkinter.ttk as ttk
from tkinter.constants import *
from PIL import Image, ImageTk
import glob
import Photometry 
import os
import shutil

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
        
        # VerticalCanvasWidth, VerticalCanvasHeight = 100,240
        # self.VerticalCanvas = tk.Canvas(Frame1, 
                # width=VerticalCanvasWidth, height=VerticalCanvasHeight, 
                # background='white')
        # self.VerticalCanvas.grid(row=1,
                # column=BlankCanvasWidth+CanvasWidth+1, 
                # rowspan=VerticalCanvasHeight, columnspan=VerticalCanvasWidth)

        ThresholdSliderLength,ThresholdSliderHeight = 360,15
        self.ThresholdSlider = tk.Scale(Frame1, from_=0, to=100, 
                orient=HORIZONTAL, label='Threshold', sliderlength = 60, 
                length=ThresholdSliderLength)
        self.ThresholdSlider.grid(row=CanvasHeight+1, column=1, 
                columnspan=ThresholdSliderLength, rowspan=ThresholdSliderHeight)

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

        # img = tk.PhotoImage(file=path)
        # self.label = tk.Label(Frame1,image=img,width=300,height=300)
        # self.label.image = img
        # self.label.grid(row=400,column=300)

        # HorizCanvasWidth,HorizCanvasHeight = 240,100
        # self.HorizCanvas = tk.Canvas(Frame1, width=HorizCanvasWidth, 
                # height=HorizCanvasHeight, background='white')
        # self.HorizCanvas.grid(row=VerticalCanvasHeight+1,column=511, 
                # rowspan=HorizCanvasHeight, columnspan=HorizCanvasWidth)
        

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

        self.openButton = tk.Button(Frame1, text='OPEN', width=7, 
                height=1, command=self.openFile)
        self.openButton.grid(row=0, column=1)

        self.QuitButton = tk.Button(Frame1, text="QUIT", 
                command=Frame1.quit, width=7)
        self.QuitButton.grid(row=0,column=3)

        # self.Text = tk.Button(Frame1, text=str(fname),command = open)
        # self.Text.grid(row=0,column=400)

        self.FolderName = tk.Entry(Frame1)
        self.FolderName.insert(0,'Event')
        self.FolderName.grid(row=0,column=5)
        
        self.CreateButton = tk.Button(Frame1, text='CREATE', width=7, 
                height=1, command=self.callback)
        self.CreateButton.grid(row=0, column=6)

    # def open():
        # Filename = fname

        # self.RunButton = tk.Button(Frame1, text="RUN",width=7)
        # self.RunButton.grid(row=0,column=2)

    def openFile(self):
        global fname
        fname = tk.filedialog.askopenfilename(title='Choose a file')
        print(FolderDirectory)
        Photometry.main(fname,FolderDirectory)
        lightcurvepath= f'/home/luke/{FolderDirectory}/LightCurve.png'
        self.OG = Image.open(lightcurvepath)
        ResizedOG = self.OG.resize((340,240),Image.ANTIALIAS)
        self.IMG = ImageTk.PhotoImage(ResizedOG)
        self.LightCurve.config(image=self.IMG) 
        MaxFrameNumber= len(glob.glob("ObjectPlot*.png"))
        self.FrameSlider.config(to=MaxFrameNumber)

    def FrameValue(self,event):
        FrameNumber = self.var.get()
        path= f'/home/luke/{FolderDirectory}/ObjectPlot{FrameNumber:03}.png'
        self.Original = Image.open(path)
        Resized = self.Original.resize((340,340),Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(Resized)
        self.display.config(image=self.img)

    def callback(self):
        global FolderDirectory
        FolderDirectory = (self.FolderName.get())
        FolderPath = f'/home/luke/{FolderDirectory}'
        if os.path.exists(FolderPath):
            shutil.rmtree(FolderPath)
        os.makedirs(FolderPath)
        print(FolderPath)

if __name__ == '__main__':
    GUI.main()
