# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:10:59 2021

@author: styra
"""

from tkinter import *
 
class CanvasDemo:
    def __init__(self):
        window = Tk()
        window.title("CanvasDemo")
 
        self.canvas = Canvas(window, width = 200, height = 100, bg = "White")
        self.canvas.pack()
 
        frame = Frame(window)
        frame.pack()
 
        btRectangle = Button(frame, text = "长方形", command = self.displayRect)
        btOval = Button(frame, text="椭 圆", command=self.displayOval)
        btArc = Button(frame, text = "圆 弧", command = self.displayArc)
        btPolygon = Button(frame, text="多边形", command=self.displayPolygon)
        btLine = Button(frame, text=" 线 ", command=self.displayLine)
        btString = Button(frame, text="文 字", command=self.displayString)
        btClear = Button(frame, text="清 空", command=self.clearCanvas)
 
        btRectangle.grid(row = 1, column = 1)
        btOval.grid(row=1, column=2)
        btArc.grid(row=1, column=3)
        btPolygon.grid(row=1, column=4)
        btLine.grid(row=1, column=5)
        btString.grid(row=1, column=6)
        btClear.grid(row=1, column=7)
 
        window.mainloop()
 
    def displayRect(self):
        self.canvas.create_rectangle(10, 10, 190, 90, tags = "rect")
    def displayOval(self):
        self.canvas.create_oval(10, 10, 190, 90, tags = "oval", fill = "red")
    def displayArc(self):
        self.canvas.create_arc(10, 10, 190, 90, start = -90, extent = 90, width = 5, fill = "red", tags = "arc")
    def displayPolygon(self):
        self.canvas.create_polygon(10, 10, 190, 90, 30, 50, tags = "polygon")
    def displayLine(self):
        self.canvas.create_line(10, 10, 190, 90, fill = 'red', tags = "line")
        self.canvas.create_line(10, 90, 190, 10, width = 9, arrow = "last", activefill = "blue", tags = "line")
    def displayString(self):
        self.canvas.create_text(60, 40, text = "Hi,i am a string", font = "Tine 10 bold underline", tags = "string")
    def clearCanvas(self):
        self.canvas.delete("rect", "oval", "arc", "polygon", "line", "string")
 
CanvasDemo()