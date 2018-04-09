import tkinter as tk
from tkinter.filedialog import askopenfilename
import tkinter.ttk as ttk
from tkinter.constants import *
from PIL import Image, ImageTk

class GUI(tk.Frame):
    
    @classmethod
    def main(cls):
        root = tk.Tk()
        app = cls(root)
        root.geometry('800x500+50+50')
        root.mainloop()

    def __init__(self, root):
        super().__init__(root)
        self.master = root

        self.tk_setPalette( background = 'gray10',
                            foreground = 'gray80',
                            activeBackground = 'gray30',
                            selectColor = 'darkorange3',
                            selectBackground = 'darkorange4' )

        self.createwidgets()

    def createwidgets(self):
        self.pack(fill=BOTH, expand = True)

        frame1 = tk.Frame(self)
        frame1.pack(fill=Y)
        
        self.TheCanvas = tk.Canvas(frame1, width=360, height=240, background='darkorange3')
        self.TheCanvas.pack( side = LEFT )

        self.TheBlankCanvas = tk.Canvas(frame1, width=50, height=240)
        self.TheBlankCanvas.pack( side = LEFT ) 

        self.TheHeightCanvas = tk.Canvas(frame1, width=100, height=240, background='white')
        self.TheHeightCanvas.pack( side = LEFT )

        frame2 = tk.Frame(self)
        frame2.pack(side = LEFT)
        self.TheThresholdSlider = tk.Scale(frame2, from_=0, to=100, orient=HORIZONTAL, label='Treshold', sliderlength = 60, length=360)
        self.TheThresholdSlider.pack( side = TOP )
       
        self.TheFrameSlider = tk.Scale(frame2, from_=0, to=90, orient=HORIZONTAL, label='Frame', sliderlength = 60, length = 360)
        self.TheFrameSlider.pack( side = BOTTOM ) 

        frame3= tk.Frame(self, width = 100)
        frame3.pack(side = RIGHT)
        v = tk.IntVar()
        self.TheSelection = tk.Radiobutton(frame3, text="Object", variable=v, value=0,indicatoron=0, width = 12)
        self.TheSelection2 = tk.Radiobutton(frame3, text="Reference Star", variable=v, value=1,indicatoron=0, width=12)
        self.TheSelection.pack( side = TOP)
        self.TheSelection2.pack( side = BOTTOM)


        # self.TheWidthCanvas = tk.Canvas(cntl_frame, width=240, height=100, background='white')
        # self.TheWidthCanvas.grid(row=241,column=511, rowspan=100, columnspan=240)

        # self.TheLightCanvas = tk.Canvas(cntl_frame, width=340, height=170, background='white')
        # self.TheLightCanvas.grid(row=350,column=411, rowspan=170, columnspan=340)

        # self.openButton = tk.Button(text='Open File', width=10, height=1, command=self.openFile)
        # self.openButton.grid(row=1000, column=10,rowspan=10, columnspan=1)



        # self.thePic = tk.Label(image='', bg='black')
        # self.thePic.grid(row=1, column=0, padx=5, pady=5)



    def testbutton(self):
        # self.TheButton.config(text='Goodbye')
        if self.TheButton['text'] == 'Hello':
            self.TheButton['text'] = 'Goodbye'
            self.TheButton['bg'] = 'darkorange3'
        else:
            self.TheButton['text'] = 'Hello'

    def openFile(self):
        fname = tk.filedialog.askopenfilename(title='Choose a file')

if __name__ == '__main__':
    GUI.main()
