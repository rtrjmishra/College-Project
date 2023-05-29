from PIL import Image, ImageTk
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import tkinter.font as font
from functools import partial
import tensorflow as tf
import cv2
import numpy as np
import os
import threading
import time
from functools import partial
import queue
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, LeakyReLU
import functools

class ResInResDenseBlock(tf.keras.layers.Layer):
    """Residual in Residual Dense Block"""
    def __init__(self, nf=64, gc=32, res_beta=0.2, wd=0., name='RRDB',
                 **kwargs):
        super(ResInResDenseBlock, self).__init__(name=name, **kwargs)
        self.res_beta = res_beta
        self.rdb_1 = ResDenseBlock_5C(nf, gc, res_beta=res_beta, wd=wd)
        self.rdb_2 = ResDenseBlock_5C(nf, gc, res_beta=res_beta, wd=wd)
        self.rdb_3 = ResDenseBlock_5C(nf, gc, res_beta=res_beta, wd=wd)
        
        
    def call(self, x):
        out = self.rdb_1(x)
        out = self.rdb_2(out)
        out = self.rdb_3(out)
        return out * self.res_beta + x

class ResDenseBlock_5C(tf.keras.layers.Layer):
    """Residual Dense Block"""
    def __init__(self, nf=64, gc=32, res_beta=0.2, wd=0., name='RDB5C',**kwargs):
        super(ResDenseBlock_5C, self).__init__(name=name, **kwargs)
        # gc: growth channel, i.e. intermediate channels
        self.res_beta = res_beta
        lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
        _Conv2DLayer = functools.partial(
            Conv2D, kernel_size=3, padding='same',
            kernel_initializer=_kernel_init(0.1), bias_initializer='zeros',
            kernel_regularizer=_regularizer(wd))
        self.conv1 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv2 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv3 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv4 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv5 = _Conv2DLayer(filters=nf, activation=lrelu_f())

    def call(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(tf.concat([x, x1], 3))
        x3 = self.conv3(tf.concat([x, x1, x2], 3))
        x4 = self.conv4(tf.concat([x, x1, x2, x3], 3))
        x5 = self.conv5(tf.concat([x, x1, x2, x3, x4], 3))
        return x5 * self.res_beta + x
    #custom_objects={'ResInResDenseBlock': ResInResDenseBlock,'ResDenseBlock_5C':ResDenseBlock_5C,"LeakyReLU":LeakyReLU}
def _kernel_init(scale=1.0, seed=None):
    """He normal initializer with scale."""
    scale = 2. * scale
    return tf.keras.initializers.VarianceScaling(
        scale=scale, mode='fan_in', distribution="truncated_normal", seed=seed)
def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)

class page1(tk.Frame):
    def __init__(self, parent, file_path, width, height):
        super(page1, self).__init__(parent, borderwidth=0, highlightthickness=0)
        self.width=width
        self.height=height
        self.canvas = tk.Canvas(self, width=width, height=height)
        self.canvas.pack()
        pil_img = Image.open(file_path)
        self.img = ImageTk.PhotoImage(pil_img.resize((width, height), Image.ANTIALIAS))
        self.bg = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
        myFont5 = font.Font(family='Aleo', size=40, weight='bold')
        self.text =self.canvas.create_text( 1090,300, text = "Model",font=myFont5 ,fill="#d6c4d6")
        self.text =self.canvas.create_text( 1090, 470, text ="Scaling factor" ,font=myFont5, fill="#d6c4d6")
        self.Imagepath = None
        self.loadingtext=None
        self.padding_width =0
        self.padding_height=0
        self.pageFrame=None
        self.ModelSelect=None
        self.MenuModel=None
        self.t1=None
        self.t1=None
        self.Menufactor=None
        self.label1=None
    def page1content(self,frame,Aboutframe):
        self.pageFrame=frame
        FontAbout = font.Font(family='Aleo', size=15, weight='bold')
        frame2_btn = tk.Button(frame, text='About',fg='#050a28',font=FontAbout, bg='#d6c4d6',  activebackground='#ccbcce',
                                   activeforeground="#7e3573", bd=0, cursor="hand2",
                                command=lambda: self.show_About_frame(Aboutframe), width=10, height=1).place(x=1230, y=15)
        myFont = font.Font(family='Courier', size=15, weight='bold')
        #photo = tk.PhotoImage(file =r"upload.png")

        # Resizing image to fit on button
        #photoimage = photo.subsample(20, 20)
        self.uploadButton=tk.Button(frame, text = '   Select Image   ', bg='#ccbcce',fg='#050a28',activebackground='#050a28', width=29, height=1,
                                   activeforeground="#ccbcce",bd=0, cursor="hand2", command = lambda:self.upload_file(self.uploadButton,frame))
        #to add image but image = photoimage,compound = "left",
        self.uploadButton['font'] = myFont
        self.uploadButton.place(x=90,y=215)
        #image = photoimage,compound = "left",
        #b1 =tk.Button(frame1, text='Upload File', image = photoimage, compound = "left", cursor="hand2",command = lambda:upload_file()).place(x=120, y=400)
        #b2 =tk.Button(frame1, text = 'Click Me !', image = photoimage, compound = "left").place(x=620,y=500)
        
        
        # Dropdown menu options model
        options = ["              SRCNN              "   ,"              RFDN               "   ,"              Autoencoder           " ]
        # datatype of menu text
        self.MenuModel = tk.StringVar()
        # initial menu text

        
        self.MenuModel.set( "RFDN" )
        # Create Dropdown menu
        drop = tk.OptionMenu( frame , self.MenuModel , *options )
        #function OptionMenu frame , clicked= stringvar click of array ,  pointer in array
        myFont1 = font.Font(family='Aleo', size=20, weight='bold')
        myFont2= font.Font(family='Aleo', size=15, weight='bold')
        drop.configure(fg='#050a28', bg='#d6c4d6',activebackground='#050a28',font=myFont1 ,width=20, height=1,highlightthickness=0,relief="flat",
                       activeforeground="#d6c4d6",bd=0, cursor="hand2")
        drop["menu"].configure( bg='#d6c4d6',fg='#050a28',activebackground='#050a28',activeborderwidth=5,font=myFont2,relief="flat",
                                       activeforeground="#d6c4d6",bd=0, cursor="hand2")
        # borderwidth=0
        drop.place(x=930,y=350)

        
        #Dropdown menu options X Factor
        options = ["                     X2                     "]
        # datatype of menu text
        self.Menufactor = tk.StringVar()
        # initial menu text
        self.Menufactor.set( "X2" )
        
        frame.update()
        # Create Dropdown menu
        drop = tk.OptionMenu( frame , self.Menufactor , *options )
        #function OptionMenu frame , clicked= stringvar click of array ,  pointer in array
        myFont1 = font.Font(family='Aleo', size=20, weight='bold')
        myFont2= font.Font(family='Aleo', size=15, weight='bold')
        drop.configure(fg='#050a28', bg='#d6c4d6',activebackground='#050a28',font=myFont1 ,width=20, height=1,highlightthickness=0,
                       activeforeground="#d6c4d6",bd=0, cursor="hand2")
        drop["menu"].configure( bg='#d6c4d6',fg='#050a28',activebackground='#050a28',activeborderwidth=5,font=myFont2,
                                       activeforeground="#d6c4d6",bd=5, cursor="hand2")
        drop.place(x=930,y=510)
        #frame.after(500, self.update_the_OptionMenu)
        self.MenuModel.trace("w",self.update_the_OptionMenu)
        
        s = ttk.Style()
        s.theme_use('alt')
        s.configure("red.Horizontal.TProgressbar", troughcolor ='#d6c4d6', background='#050a28')
        self.bar=ttk.Progressbar(frame, style="red.Horizontal.TProgressbar", orient="horizontal", length=400, mode="determinate")
         
        myFont1 = font.Font(family='Aleo', size=28, weight='bold')
        self.Startbutton = tk.Button(frame, text='Start', fg='#050a28', bg='#d6c4d6', width=12, height=1,activebackground='#050a28',
                                   activeforeground="#ccbcce", bd=0, cursor="hand2", command=lambda: self.show_frame2(self.Imagepath, self.MenuModel.get(),self.Menufactor.get(),frame,self.Startbutton))
        self.Startbutton['font'] = myFont1
        self.Startbutton.place(x=550,y=600)
        
    def update_the_OptionMenu(self,*args):
        if((self.MenuModel.get()).strip()=="RFDN"):
            self.pageFrame.update()
            options = ["                     X2                     "]
            self.Menufactor = tk.StringVar()
            # initial menu text
            self.Menufactor.set( "X4" )

            self.pageFrame.update()
            # Create Dropdown menu
            drop = tk.OptionMenu( self.pageFrame , self.Menufactor , *options )
            #function OptionMenu frame , clicked= stringvar click of array ,  pointer in array
            myFont1 = font.Font(family='Aleo', size=20, weight='bold')
            myFont2= font.Font(family='Aleo', size=15, weight='bold')
            drop.configure(fg='#050a28', bg='#d6c4d6',activebackground='#050a28',font=myFont1 ,width=20, height=1,highlightthickness=0,
                           activeforeground="#d6c4d6",bd=0, cursor="hand2")
            drop["menu"].configure( bg='#d6c4d6',fg='#050a28',activebackground='#050a28',activeborderwidth=5,font=myFont2,
                                           activeforeground="#d6c4d6",bd=5, cursor="hand2")
            drop.place(x=930,y=510)
            self.pageFrame.update()
        else:
            options = ["                     X2                     "]
            # datatype of menu text
            self.Menufactor = tk.StringVar()
            # initial menu text
            self.Menufactor.set( "X2" )

            self.pageFrame.update()
            # Create Dropdown menu
            drop = tk.OptionMenu( self.pageFrame , self.Menufactor , *options )
            #function OptionMenu frame , clicked= stringvar click of array ,  pointer in array
            myFont1 = font.Font(family='Aleo', size=20, weight='bold')
            myFont2= font.Font(family='Aleo', size=15, weight='bold')
            drop.configure(fg='#050a28', bg='#d6c4d6',activebackground='#050a28',font=myFont1 ,width=20, height=1,highlightthickness=0,
                           activeforeground="#d6c4d6",bd=0, cursor="hand2")
            drop["menu"].configure( bg='#d6c4d6',fg='#050a28',activebackground='#050a28',activeborderwidth=5,font=myFont2,
                                           activeforeground="#d6c4d6",bd=5, cursor="hand2")
            drop.place(x=930,y=510)
            self.pageFrame.update()
            
    def upload_file(self,uploadButton,frame):
        global img
        f_types = [('Jpg Files', '*.jpg'),('PNG file','*.png')]
        self.Imagepath = filedialog.askopenfilename(filetypes=f_types)
        if (self.Imagepath !=""):
            if(self.label1!=None):
                self.label1.place_forget()
                self.pathimage.place_forget()
                self.Label_image_size.place_forget()
                self.pageFrame.update()
            
            
            myFont = font.Font(family='Courier', size=15, weight='bold')
            #photo = tk.PhotoImage(file =r"upload.png")
            self.uploadButton['text'] = 'Edit Image'
            # Resizing image to fit on button
            self.pageFrame.update()
            myFont2 = font.Font(family='Aleo', size=18)
            
            imageName=os.path.basename(os.path.normpath(self.Imagepath)) #to get name of the image
            self.pathimage=tk.Label(frame1, text= imageName,bg='#050a28',fg='#fff',font=myFont2)
            self.pathimage.place(x=200,y=540)
            image1 = Image.open(self.Imagepath)
            
            image1 = image1.resize((328, 265), Image.ANTIALIAS)
            
            test = ImageTk.PhotoImage(image1)
            
            self.label1 = tk.Label(frame1,image=test,width="328",height="265",bd=0)
            
            self.label1.image = test
            
            self.label1.place(x=100,y=265)  # image
            image=cv2.imread(self.Imagepath)
            height, width, channels = image.shape
            self.Label_image_size=tk.Label(frame1, text= "( "+str(width)+" , "+str(height)+" )",bg='#050a28',fg='#fff',font=myFont2)
            self.Label_image_size.place(x=200,y=575)
            self.pageFrame.update()
            
    def show_frame2(self,filepath,Model,xfactor,frame,Continuebutton):
        if(filepath==None or filepath==""):
            myFont2 = font.Font(family='Aleo', size=18)
            label_validation=tk.Label(frame, text= "Please Select The Image ",bg='#050a28',fg='#FF0000',font=myFont2)
            label_validation.place(x=550,y=300)
            frame.update()
            time.sleep(0.6)
            label_validation.place_forget()
        else:

            self.Startbutton.place_forget()
            self.bar.place(x=500,y=650)
            #Continuebutton['text'] = 'new value'

            frame.update()
            q = queue.Queue()
            handler = partial(self.on_update, q=q, pb=self.bar)

            # Регистрируем обработчик для события обновления progressbar'а
            frame.bind('<<Updated>>', handler)
            #myFont5 = font.Font( family  =  'Aleo', size=40)
            #self.loadingtext=self.canvas.create_text( 675 , 640, text =" loading ... " ,font=myFont5, fill="#d6c4d6")
            #frame.update()


            self.t2 = threading.Thread( target=self.RunTrain, args=(filepath,Model,xfactor,frame,q))
            self.t1 = threading.Thread( target=self.progress_bar, args=(q, frame))

            # starting thread 1
            self.t1.start()
            # starting thread 2
            self.t2.start()
        #time.sleep(1)
        #superResolution,lowResolution = self.Models(filepath, Model, xfactor)
        #hsuper,wSuper,c=superResolution.shape
        #hLow,Wlow,c=lowResolution.shape
            
    def RunTrain(self,filepath,Model,xfactor,frame,q):
        superResolution,lowResolution = self.Models(filepath, Model, xfactor)
        hsuper,wSuper,c =superResolution.shape
        hLow,Wlow,c= lowResolution.shape
        frame3 = page3(window, self.width ,self.height,Model,Wlow,hLow,wSuper,hsuper)
        frame3.grid(row = 0, column = 0, sticky='nsew')
        page3.page2content(frame3,filepath,Model,xfactor,superResolution,frame)
        frame3.tkraise()
        
        self.Startbutton.place(x=550,y=600)
        self.bar.place_forget()
        self.label1.place_forget()
        self.pathimage.place_forget()
        self.Imagepath=None
        self.Label_image_size.place_forget()
        self.uploadButton['text'] = ' Select Image '
        self.pageFrame.update()
        
        
    def progress_bar(self,q, r):
        for i in range(100):
            q.put(i + 1)
            r.event_generate('<<Updated>>', when='tail')
            time.sleep(0.1)


    def on_update(self,event, q=None, pb=None):
        self.bar['value'] = q.get()

    def Models(self,filepath,ModelName,Xfactor):
        sr1=[]
        imagelow=None
        if(ModelName.strip()=="SRCNN"):
            if(Xfactor.strip()=="X2"):
                SRCNN = tf.keras.models.load_model('./SRCNN/srcnn_model_2x_90epoch.h5')
                imagelow=cv2.imread(filepath)
                sr1=self.SRCNN_model_2x(imagelow,SRCNN)

            elif(Xfactor.strip()=="X4"):
                SRCNN = tf.keras.models.load_model('./SRCNN/srcnn_model_4x_65epoch.h5')
                image=cv2.imread(filepath)
                imagelow = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                sr1=self.SRCNN_model_4x(imagelow,SRCNN)
        elif (ModelName.strip() =="RFDN"):
            image=cv2.imread(filepath)
            imagelow = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image= imagelow.astype(np.float32)/255
            image=[image]
            image=np.array(image)
            if(Xfactor.strip()=="X2"):
                RFDN_x2 = tf.keras.models.load_model('./RFDN/RFDN_x2.h5')
                sr1 = np.clip(RFDN_x2.predict(image), 0.0, 1.0)
            elif(Xfactor.strip()=="X4"):
                RFDN_x4 = tf.keras.models.load_model('./RFDN/RFDN_x4.h5')
                sr1 = np.clip(RFDN_x4.predict(image), 0.0, 1.0)
            
            sr1=sr1[0]
        elif(ModelName.strip()=="Autoencoder"):
            if(Xfactor.strip()=="X2"):
                autoencoder2x = tf.keras.models.load_model('./Autoencoder/AutoencoderX2.h5')
                image=cv2.imread(filepath)
                imagelow = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width, channels = imagelow.shape
                Imagetest,padding_width,padding_height =self.testSize(imagelow,height, width, channels,autoencoder2x)
                sr1=self.deletepadding(Imagetest[0],padding_width*2,padding_height*2)
            elif(Xfactor.strip()=="X4"):
                autoencoder4x = tf.keras.models.load_model('./Autoencoder/Autoencoder_42ep_Subpixel_model.h5')
                image=cv2.imread(filepath)
                imagelow = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width, channels = imagelow.shape
                Imagetest,padding_width,padding_height =self.testSize(imagelow,height, width, channels,autoencoder4x)
                sr1=self.deletepadding(Imagetest[0],padding_width*4,padding_height*4)
        return sr1,imagelow
    
    def show_About_frame(self,frame):
        frame.tkraise()
        
#--------------------------------------------------------------------------------------------------
#autoencoder make padding to the input
    def checkSize(self,height):
        if (height % 2) == 0:
            height=int(height/2)
            if (height % 2) == 0:
                height=int(height/2)
                if (height % 2) == 0:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
        return False
    
    def paddingHeight(self,img, H, W, C):
        zimg = np.zeros((H+2, W, C))
        zimg[1:H+1, :W, :C] = img
        
        # Pad the first/last two col and row
        zimg[H+1:H+2, :W, :] = img[H-1:H , :W, :]
        zimg[0:1, :W,:C] = img[0:1, :W, :C]
        return zimg

    def paddingWidth(self,img, H, W, C):
        zimg = np.zeros((H, W+2, C))
        zimg[:H ,1:W+1, :C] = img
        # Pad the first/last two col and row
        zimg[ :H,W+1:W+2, :] = img[ :H,W-1:W , :]
        zimg[ :H,0:1,:C] = img[:H,0:1, :C]
        return zimg
    
    def deletepadding(self,image,padding_width,padding_height):
        global Img
        if(padding_width ==0):
            if(padding_height ==0):
                Img=image
                return Img
            else:
                
                Img=image[padding_height:image.shape[0]-padding_height,:image.shape[1],:image.shape[2]]

                return Img
            
        else:
            
            imageRemoveWidth=image[:image.shape[0] ,padding_width:image.shape[1]-padding_width,:image.shape[2]]

            self.deletepadding(imageRemoveWidth,0,padding_height)
        return Img
    
    def testSize(self,imagelow,height, width, channels,autoencoder_model):
        if( self.checkSize(height)):
            if( self.checkSize(width)):
                image= imagelow.astype(np.float32)/255
                image=[image]
                image=np.array(image)
                global sr1
                sr1 = np.clip(autoencoder_model.predict(image), 0.0, 1.0)
            else:
                image=self.paddingWidth(imagelow,height, width, channels)
                
                self.padding_width =self.padding_width+1
                height, width, channels = image.shape
                self.testSize(image,height, width, channels,autoencoder_model)
        else:
            image=self.paddingHeight(imagelow,height, width, channels)
            self.padding_height=self.padding_height+1
            height, width, channels = image.shape
            self.testSize(image,height, width, channels,autoencoder_model)
        return sr1,self.padding_width,self.padding_height
    
    def One_channel_test_2x(self,fullimg,Model):
        upscale_factor=2
        width = fullimg.shape[0]
        height = fullimg.shape[1]
        img = fullimg

        floatimg = img.astype(np.float32) / 255.0
        imgYCbCr = cv2.cvtColor(floatimg, cv2.COLOR_RGB2YCrCb)
        imgY = imgYCbCr[:, :, 0]
        LR_input_ = imgY.reshape(1, imgY.shape[0], imgY.shape[1], 1)
        Y = Model.predict([LR_input_])[0]
        Cr = np.expand_dims(cv2.resize(imgYCbCr[:, :, 1], None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC),
                            axis=2)
        Cb = np.expand_dims(cv2.resize(imgYCbCr[:, :, 2], None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC),
                            axis=2)
        HR_image_YCrCb = np.concatenate((Y, Cr, Cb), axis=2)
        HR_image = ((cv2.cvtColor(HR_image_YCrCb, cv2.COLOR_YCrCb2RGB))).clip(min=0, max=1)
        #HR_image = (HR_image).astype(np.uint8)
        return HR_image
# ESPCN ( test one channel)
    def One_channel_test_4x(self,fullimg,Model):
        upscale_factor=4
        width = fullimg.shape[0]
        height = fullimg.shape[1]
        img = fullimg
        floatimg = img.astype(np.float32) / 255.0
        imgYCbCr = cv2.cvtColor(floatimg, cv2.COLOR_RGB2YCrCb)
        imgY = imgYCbCr[:, :, 0]
        LR_input_ = imgY.reshape(1, imgY.shape[0], imgY.shape[1], 1)
        Y = Model.predict([LR_input_])[0]
        Cr = np.expand_dims(cv2.resize(imgYCbCr[:, :, 1], None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC),
                            axis=2)
        Cb = np.expand_dims(cv2.resize(imgYCbCr[:, :, 2], None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC),
                            axis=2)
        HR_image_YCrCb = np.concatenate((Y, Cr, Cb), axis=2)
        HR_image = ((cv2.cvtColor(HR_image_YCrCb, cv2.COLOR_YCrCb2RGB))).clip(min=0, max=1)
        #HR_image = (HR_image).astype(np.uint8)
        return HR_image
    def SRCNN_model_2x(self,fullimg,Model):
        #SRCNN------------------------------------------------------------
        upscale_factor=2
        width = fullimg.shape[0]
        height = fullimg.shape[1]
        img = fullimg

        floatimg = img.astype(np.float32) / 255.0
        imgYCbCr = cv2.cvtColor(floatimg, cv2.COLOR_BGR2YCrCb)
        imgY = imgYCbCr[:, :, 0]
        imgY = np.expand_dims(cv2.resize(imgYCbCr[:, :, 0], None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC),
                            axis=2)

        LR_input_ = imgY.reshape(1, imgY.shape[0], imgY.shape[1], 1)
        Y = Model.predict([LR_input_])[0]
        Cr = np.expand_dims(cv2.resize(imgYCbCr[:, :, 1], None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC),
                            axis=2)
        Cb = np.expand_dims(cv2.resize(imgYCbCr[:, :, 2], None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC),
                            axis=2)
        HR_image_YCrCb = np.concatenate((Y, Cr, Cb), axis=2)
        HR_image = ((cv2.cvtColor(HR_image_YCrCb, cv2.COLOR_YCrCb2RGB))).clip(min=0, max=1)
        return HR_image
    def SRCNN_model_4x(self,fullimg,Model):
        #SRCNN------------------------------------------------------------
        upscale_factor=4
        width = fullimg.shape[0]
        height = fullimg.shape[1]
        img = fullimg

        floatimg = img.astype(np.float32) / 255.0
        imgYCbCr = cv2.cvtColor(floatimg, cv2.COLOR_BGR2YCrCb)
        imgY = imgYCbCr[:, :, 0]
        imgY = np.expand_dims(cv2.resize(imgYCbCr[:, :, 0], None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC),
                            axis=2)
        
        LR_input_ = imgY.reshape(1, imgY.shape[0], imgY.shape[1], 1)
        Y = Model.predict([LR_input_])[0]
        Cr = np.expand_dims(cv2.resize(imgYCbCr[:, :, 1], None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC),
                            axis=2)
        Cb = np.expand_dims(cv2.resize(imgYCbCr[:, :, 2], None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC),
                            axis=2)
        HR_image_YCrCb = np.concatenate((Y, Cr, Cb), axis=2)
        HR_image = ((cv2.cvtColor(HR_image_YCrCb, cv2.COLOR_YCrCb2BGR))).clip(min=0, max=1)
        return HR_image


class page3(tk.Frame):
    def __init__(self, parent, width, height,ModelName,smallsizeW=0,smallsizeH=0,supersizeW=0,supersizeH=0):
        super(page3, self).__init__(parent, borderwidth=0, highlightthickness=0)
        self.canvas = tk.Canvas(self, width=width, height=height)
        self.canvas.pack()
        pil_img = Image.open("imageBackground.png")
        self.img = ImageTk.PhotoImage(pil_img.resize((width, height), Image.ANTIALIAS))
        self.bg = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
        self.text =self.canvas.create_text( 300,100, text = "Your Image",font=("coustard", 38) ,fill="#d6c4d6")
        self.text =self.canvas.create_text( 1030, 100,text ="Super Resolution" ,font=("coustard", 38), fill="#d6c4d6")
        self.text =self.canvas.create_text( 640, 180, text ="Image Size" ,font=("coustard", 32), fill="#d6c4d6")
        self.text =self.canvas.create_text( 635, 280, text =str(smallsizeW)+"*"+str(smallsizeH)+">"+str(supersizeW)+"*"+str(supersizeH),font=("coustard", 30), fill="#d6c4d6")
        self.text =self.canvas.create_text( 642, 350, text =ModelName.strip()+" Model" ,font=("coustard", 30), fill="#d6c4d6")
    @staticmethod
    def page2content(frame,filepath, Model, xfactor, SuperResolutionImage,prevframe):

        image1 = Image.open(filepath)
        image1 = image1.resize((200, 200), Image.ANTIALIAS)
        test = ImageTk.PhotoImage(image1)
        label1 = tk.Label(frame,image=test,width="200",height="200",bd=0)
        label1.image = test
        label1.place(x=200, y=200)

        #image2 = Image.open("Nature2.jpg")
        #image1=ImageTk.PhotoImage(image=)
        Image_array = SuperResolutionImage* 255
        Image_array = Image_array.astype(np.uint8)
        imgsuper=cv2.resize(Image_array,(300,300))
    
        img =  ImageTk.PhotoImage(image=Image.fromarray(imgsuper))
        #image2 = image2.resize((300, 300))
        #test1 = ImageTk.PhotoImage(image2)
        label2 = tk.Label(frame,image=img,width="300",height="300",bd=0)
        label2.image = img
        label2.place(x=880, y=150)
        
        
        download_image = Image.fromarray(Image_array)
        #global
        myFont3 = font.Font(family='Aleo', size=25, weight='bold')

        Continuebutton = tk.Button(frame, text=' Download ', fg='#050a28', bg='#d6c4d6', width=15, height=1,activebackground='#050a28',
                                   activeforeground="#ccbcce", bd=0, cursor="hand2", command=lambda: page3.dawnloadImage(download_image))
        Continuebutton['font'] = myFont3
        Continuebutton.place(x=500,y=530)
        
        myFont3 = font.Font(family='Aleo', size=25, weight='bold')
        AddNewImagebutton = tk.Button(frame, text=' Select other image ', fg='#050a28', bg='#d6c4d6',
                                      width=15, height=1,activebackground='#050a28',
                                   activeforeground="#ccbcce", bd=0, cursor="hand2",
                                      command=lambda: page3.show_frame(prevframe))
        AddNewImagebutton['font'] = myFont3
        AddNewImagebutton.place(x=500,y=620)
        
    def dawnloadImage(image):
        Files = [('Image', '*.jpg')]
        filename = filedialog.asksaveasfile(mode='w', defaultextension=Files,filetypes = Files)
        if not filename:
            return
        image.save(filename)
        
    
    def show_frame(frame):
        frame.tkraise()
    #def newImage():


class Aboutpage(tk.Frame):
    def __init__(self, parent, width, height,frame):
        super(Aboutpage, self).__init__(parent, borderwidth=0, highlightthickness=0)
        self.canvas = tk.Canvas(self, width=width, height=height)
        self.frame=frame
        self.canvas.pack()
        pil_img = Image.open("imageAboutUS.png")
        self.img = ImageTk.PhotoImage(pil_img.resize((width, height), Image.ANTIALIAS))
        self.bg = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
        #photo = tk.PhotoImage(file =r"upload.png")
        #image=photo,
        FontAbout = font.Font(family='Aleo', size=15, weight='bold')
        self.but =tk.Button(self, text= "back", fg='#050a28',font=FontAbout, bg='#d6c4d6',activebackground='#050a28' , activeforeground="#ccbcce", bd=0, cursor="hand2", command= self.mainframe).place(x=10,y=10)
    def mainframe(self):
        self.frame.tkraise()


window = tk.Tk()
window.state('normal')
window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)
#change App text
window.title("Super resolution")
#set Dimentions

window.resizable(False, False)  # This code helps to disable windows from resizing

HEIGTH = 700
WIDTH = 1365

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

x_cordinate = int((screen_width/2) - (WIDTH/2))
y_cordinate = int((screen_height/2) - (HEIGTH/2))



window.geometry("{}x{}+{}+{}".format(WIDTH, HEIGTH, x_cordinate-7, y_cordinate-30))
#window.geometry('{}x{}'.format(WIDTH, HEIGTH))
p1 = ImageTk.PhotoImage(file = 'info.png')
# Setting icon of master window
window.iconphoto(False, p1)
IMAGE_PATH="imageHome.png"

frame1 = page1(window, IMAGE_PATH, WIDTH, HEIGTH)
Aboutframe = Aboutpage(window,WIDTH, HEIGTH,frame1)
#for frame in (frame1, frame2, frame3):
    #frame.grid(row=0, column=0, sticky='nsew')
    
frame1.grid(row=0, column=0, sticky='nsew')
Aboutframe.grid(row=0, column=0, sticky='nsew')
frame1.page1content(frame1,Aboutframe)
frame1.tkraise()
window.mainloop()
