from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from ttkthemes import ThemedStyle
from tkinter.ttk import *
from GeoTrans import *
from ImageFiltering import *
import config
import numpy as np
from keras.models import load_model
import os
import cv2
import operator
from glob import glob
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

save_path = os.path.join('data', 'checkpoints', 'covid_vs_normal_vs_pneumonia_v1_2.hdf5')
model = load_model(save_path)
model1 = load_model(os.path.join('data', 'checkpoints','normal_vs_pneumonia_adam.hdf5'))

#file=numpy.zeros((700,600,3),numpy.uint8)
def loadImage():
    
    global filename
    filename = filedialog.askopenfilename()
    img = PIL.Image.open(filename)
    #file=img

    img = img.resize((600,600))
    
    tk_img = ImageTk.PhotoImage(img)
    canvas.create_image(300, 300, image=tk_img)     # (xpos, ypos, imgsrc)
    canvas.image = tk_img		# Keep reference to PhotoImage so Python's garbage collector
                                # does not get rid of it making the image dissapear
    config.current_image = img



def predict_model():
  im = cv2.resize(cv2.imread(filename), (256, 256))
  im = im * 1./255
  predicted_image = model.predict(np.expand_dims(im, axis=0), batch_size=1)
  print(predicted_image)
  predicted_image =np.squeeze(predicted_image, axis=0)
  
  print_class_from_prediction(predicted_image)

def print_class_from_prediction(predictions):
  label_predictions = {}
  for i, label in enumerate(['COVID-19','Normal','Pneumonia']):
    label_predictions[label] = predictions[i]

  sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )
  for i, class_prediction in enumerate(sorted_lps):
    print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
    if i==0:
      label1 = ttk.Label(buttons_frame,text=("%s:" % (class_prediction[0])), width=14,foreground = "red",font=("Times", 10, "bold"))
      label1.grid(row=4, column=1)
      label2 = ttk.Label(buttons_frame,text=("%.2f" % (class_prediction[1])),foreground = "red",font=("Times", 10, "bold"))
      label2.grid(row=4, column=2)
    elif i==1:
      label3 = ttk.Label(buttons_frame,text=("%s:" % (class_prediction[0])), width=14,foreground = "blue",font=("Times", 10, "bold"))
      label3.grid(row=5, column=1)
      label4 = ttk.Label(buttons_frame,text=("%.2f" % (class_prediction[1])),foreground = "blue",font=("Times", 10, "bold"))
      label4.grid(row=5, column=2)
    elif i==2:
      label5 = ttk.Label(buttons_frame,text=("%s:" % (class_prediction[0])), width=14,foreground = "green",font=("Times", 10, "bold"))
      label5.grid(row=6, column=1)
      label6 = ttk.Label(buttons_frame,text=("%.2f" % (class_prediction[1])),foreground = "green",font=("Times", 10, "bold"))
      label6.grid(row=6, column=2)
def predict_model1():
  im = cv2.resize(cv2.imread(filename), (256, 256))
  im = im * 1./255
  predicted_image = model1.predict(np.expand_dims(im, axis=0), batch_size=1)
  print(predicted_image)
  predicted_image =np.squeeze(predicted_image, axis=0)
  
  print_class_from_prediction1(predicted_image)

def print_class_from_prediction1(predictions):
  label_predictions = {}
  for i, label in enumerate(['Normal','Pneumonia']):
    label_predictions[label] = predictions[i]

  sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )
  for i, class_prediction in enumerate(sorted_lps):
    print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
    if i==0:
      label1 = ttk.Label(buttons_frame,text=("%s:" % (class_prediction[0])), width=14,foreground = "orange",font=("Times", 10, "bold"))
      label1.grid(row=19, column=1)
      label2 = ttk.Label(buttons_frame,text=("%.2f" % (class_prediction[1])),foreground = "orange",font=("Times", 10, "bold"))
      label2.grid(row=19, column=2)
    elif i==1:
      label3 = ttk.Label(buttons_frame,text=("%s:" % (class_prediction[0])), width=14,foreground = "green",font=("Times", 10, "bold"))
      label3.grid(row=20, column=1)
      label4 = ttk.Label(buttons_frame,text=("%.2f" % (class_prediction[1])),foreground = "green",font=("Times", 10, "bold"))
      label4.grid(row=20, column=2)
    




root = Tk()
root.title("Corona-Nidaan")
root.resizable(0, 0)
style = ThemedStyle(root)
style.set_theme("scidgrey")
#p1 = PhotoImage(file = 'output-onlinepngtools.png') 
  
# Setting icon of master window 
#root.iconphoto(False, 'output-onlinepngtools.png')

mainframe = ttk.Frame(root,padding="3 3 12 12")
mainframe.grid(row=0, column=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

canvas = Canvas(mainframe, width=600, height=600)
canvas.grid(row=0, column=0, rowspan=3)		# put in row 0 col 0 and span 2 rows
canvas.columnconfigure(0, weight=3)

buttons_frame = ttk.Frame(mainframe)
buttons_frame.grid(row=0, column=1, sticky=N)

load_image_button = ttk.Button(buttons_frame, text="Load Image...",command=loadImage)
load_image_button.grid(row=0, column=1, sticky=N+W+E, columnspan=2, pady=10)

#######################
# Translation Section #
#######################
ttk.Separator(buttons_frame, orient=HORIZONTAL).grid(row=1, columnspan=3, sticky=(W, E), pady=5)

ttk.Label(buttons_frame, text="Prediction Results",font=("Times", 10, "bold")).grid(row=2, column=1, columnspan=2)
ttk.Separator(buttons_frame, orient=HORIZONTAL).grid(row=3, columnspan=3, sticky=(W, E), pady=5)

'''
label1 = ttk.Label(buttons_frame, width=14).grid(row=4, column=1)
label2 = ttk.Label(buttons_frame, text="").grid(row=4, column=2)
label3 = ttk.Label(buttons_frame, text="Normal :",width=14).grid(row=5, column=1)
label4 = ttk.Label(buttons_frame, text="").grid(row=5, column=2)
label5 = ttk.Label(buttons_frame, text="Pneumonia :",width=14).grid(row=6, column=1)
label6 = ttk.Label(buttons_frame, text="").grid(row=6, column=2)'''


Predict_button = ttk.Button(buttons_frame, text="Predict V0.1",command=predict_model)

Predict_button.grid(row=7, column=1, columnspan=2, sticky=W+E, pady=5)

ttk.Separator(buttons_frame, orient=HORIZONTAL).grid(row=8, columnspan=3, sticky=(W, E), pady=5)


ttk.Label(buttons_frame, text="Classification Report",font=("Times", 10, "bold")).grid(row=9, column=1, columnspan=2)
ttk.Label(buttons_frame, text="of Corona-Nidaan",font=("Times", 10, "bold")).grid(row=10, column=1, columnspan=2)
ttk.Separator(buttons_frame, orient=HORIZONTAL).grid(row=11, columnspan=3, sticky=(W, E), pady=5)
label7 = ttk.Label(buttons_frame, text="Test Accuracy :",width=14,font=("Times", 10, "bold")).grid(row=12, column=1)
label8 = ttk.Label(buttons_frame, text="0.95",font=("Times", 10, "bold")).grid(row=12, column=2)
label9 = ttk.Label(buttons_frame, text="COVID-19 Precision :",width=14,font=("Times", 10, "bold")).grid(row=13, column=1)
label10 = ttk.Label(buttons_frame, text="0.94",font=("Times", 10, "bold")).grid(row=13, column=2)
label11 = ttk.Label(buttons_frame, text="COVID-19 Recall :",width=14,font=("Times", 10, "bold")).grid(row=14, column=1)
label12 = ttk.Label(buttons_frame, text="0.94",font=("Times", 10, "bold")).grid(row=14, column=2)
label13 = ttk.Label(buttons_frame, text="COVID-19 F1-Score :",width=14,font=("Times", 10, "bold")).grid(row=15, column=1)
label14 = ttk.Label(buttons_frame, text="0.94",font=("Times", 10, "bold")).grid(row=15, column=2)

ttk.Separator(buttons_frame, orient=HORIZONTAL).grid(row=16, columnspan=3, sticky=(W, E), pady=5)


segment_button = ttk.Button(buttons_frame, text="Predict V0.2",    command=predict_model1)

segment_button.grid(row=17, column=1, columnspan=2, sticky=W+E, pady=5)
ttk.Separator(buttons_frame, orient=HORIZONTAL).grid(row=18, columnspan=3, sticky=(W, E), pady=5)
'''
label15 = ttk.Label(buttons_frame, text="Train Accuracy:",width=14,font=("Times", 10, "bold")).grid(row=19, column=1)
label16 = ttk.Label(buttons_frame, text="0.88",font=("Times", 10, "bold")).grid(row=19, column=2)
label17 = ttk.Label(buttons_frame, text="Train Loss:",width=14,font=("Times", 10, "bold")).grid(row=20, column=1)
label18 = ttk.Label(buttons_frame, text="0.31",font=("Times", 10, "bold")).grid(row=20, column=2)'''
#label19 = ttk.Label(buttons_frame, text="Val Accuracy:",width=14).grid(row=21, column=1)
#label20 = ttk.Label(buttons_frame, text="0.96").grid(row=21, column=2)
#label21 = ttk.Label(buttons_frame, text="Val Loss:",width=14).grid(row=22, column=1)
#label22 = ttk.Label(buttons_frame, text="0.12").grid(row=22, column=2)

ttk.Separator(buttons_frame, orient=HORIZONTAL).grid(row=21, columnspan=3, sticky=(W, E), pady=5)
img1 = PIL.Image.open('output-onlinepngtools.png')
img1 = img1.resize((130,130))
tk_img1 = ImageTk.PhotoImage(img1)
#canvas.create_image(400, 300, image=tk_img)
ttk.Label(buttons_frame, image=tk_img1).grid(row=22, columnspan=3, pady=5)
copyright_symbol = u"\u00A9"
msg = u"Copyright: %s \nMainak Chakraborty \n& Dr.Sunita Dhavale, DIAT" % (copyright_symbol)
ttk.Label(buttons_frame, text=msg,foreground = "dark blue",font=("Times", 10, "bold"),).grid(row=23, columnspan=3, pady=5)

root.mainloop()