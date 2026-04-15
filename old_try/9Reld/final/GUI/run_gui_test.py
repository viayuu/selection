import sys
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinter import messagebox, filedialog
import paramiko
# import yaml
import warnings
import threading
# from ruamel.yaml.scalarstring import DoubleQuotedScalarString as DQ
# from ruamel.yaml import YAML


from PIL import Image, ImageTk


from datetime import datetime
import yaml
import re
import webbrowser
from tkinter import PhotoImage
import os
import stat
import posixpath
import ast
import matplotlib.pyplot as plt
import math
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import textwrap
import shutil
from pathlib import Path



#===============================================================================================================
# user defined parameters
# ================================================================================================================
HOSTNAME = '10.20.63.11'
PORT = 50007
USERNAME = 'luoqing'
KEY = 'E:\lq\keys\id_rsa'
PASSWORD = ''
DEVICE = 'cuda:0'
GUI_DIR = "/public/home/luoqing/code/EasyNCO_v0804/EasyNCO/GUI" 
ROOT_DIR = '/public/home/luoqing/code/EasyNCO_v0804/EasyNCO'  # 最好写成pargs 形式
PYTHON_ROUTE = '/public/home/luoqing/anaconda3/envs/EasyNCO/bin/python'
DATASET =  ROOT_DIR + 'EasyNCO/data/datasets/test_dataset_tsp_uniform/test_tsp100_nums10000_uniform.pt'
CKPT = ROOT_DIR + "EasyNCO/results/train/pomo_tsp/2025-07-08-19-14-27_pomo_tsp_100_train/checkpoints/pomo_tsp100_epoch29.ckpt"

CURRENT_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))  # '{your_path}/EasyCO'
# ================================================================================================================


METHOD_LIST = ['pomo','am','LIH','mtpomo',
               'mvmoe','dact','deepaco',
               'difusco','dpn','elg','icam','glop',
               'htsp','invit','lehd','matnet',
               'nlns','pointerformer','t2t',
               'l2s','drl_hgnn','matpoenet', 'psl']


"""TASK_LIST = ['tsp','atsp','bpp','cvrp',
             'ffsp', 'kp','mis','mkp',
             'op', 'pctsp','rcpsp','smtwtp',
             'sop',]"""
             
TASK_LIST = ['tsp', 'atsp', 'pctsp', 'cvrp', 'op', 'sop',
             'ffsp', 'rcpsp', 'smtwtp',
             'kp', 'mkp', 'bpp',
             'mis', 'sat', 'hcp', 'mtsp', 'mpdp', 'mdvrp', 'fmdvrp','jssp','fjsp', 'mvrp',
             'vrptw', 'ovrp', 'vrpl', 'ovrptw', 'ovrpb', 'vrpltw', 'vrpbl', 'ovrpbl',
             'vrpbltw', 'ovrpltw', 'ovrpl', 'ovrpbtw', 'vrpbtw', 'ovrpbltw',
             'motsp', 'mocvrp', 'mokp',
            ]

UTILS_YAML = "./utils_gui_params.yaml"

EXCLUDE_KEYS = {'env_name', 'aug_type', 'trainer', 'test_loader'}



#===============================================================================================================
# fonts 
#===============================================================================================================
WATERMARK = []

WATERMARK.append("┌─────────────────────────────────────────────┐\n")
WATERMARK.append("|  #####    #     ####  #   #   ###    ###    |\n")
WATERMARK.append("|  #       # #   #       # #   #      #   #   |\n")
WATERMARK.append("|  ####   #   #   ###     #    #      #   #   |\n")
WATERMARK.append("|  #      #####      #    #    #      #   #   |\n")
WATERMARK.append("|  #####  #   #  ####     #     ###    ###    |\n")
WATERMARK.append("└─────────────────────────────────────────────┘\n")




TEXT = []

TEXT.append(" #     #  #######  #        #          ####     ###    #     #  #######\n")
TEXT.append(" #     #  #        #        #         #        #   #   ##   ##  #\n")
TEXT.append(" #     #  #        #        #        #        #     #  # # # #  #\n")
TEXT.append(" #  #  #  #####    #        #        #        #     #  #  #  #  #####\n")
TEXT.append(" # # # #  #        #        #        #        #     #  #     #  #\n")
TEXT.append(" ##   ##  #        #        #         #        #   #   #     #  #\n")
TEXT.append(" #     #  #######  ######   ######     ####     ###    #     #  #######\n")

TEXT.append("\n")
TEXT.append("\n")

TEXT.append(" "*18+" #######     #      #####   #     #    ####     ###\n")
TEXT.append(" "*18+" #           #     #     #   #   #    #        #   #\n")
TEXT.append(" "*18+" #          ###    #          # #    #        #     #\n")
TEXT.append(" "*18+" #####      # #     #####      #     #        #     #\n")
TEXT.append(" "*18+" #         #####         #     #     #        #     #\n")
TEXT.append(" "*18+" #         #   #   #     #     #      #        #   #\n")
TEXT.append(" "*18+" #######  ##   ##   #####      #       ####     ###\n")


# ================================================================================================================


########################################################################
class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.mode = 'test'
        self.title('EasyCO')
        self.geometry('1350x900')
        self.configure(bg="white")
        
        
        # ===============================================================================================================
        # flag
        # ===============================================================================================================
        self.dataset_flag = False
        self.ckpt_flag = False
        
        # ==============================================================================================================    
        # tag_set       
        # ==============================================================================================================
        self._defined_tags = set()
        

        
        
       # ==============================================================================================================
        #  color 
        # ==============================================================================================================
        style = ttk.Style()
        style.configure("White.TFrame", background="white")
        style.configure("White.TLabel", background="white")
        style.configure("White.TEntry", fieldbackground="white", background="white")
        style.configure("White.TButton", background="white")
        style.configure('Custom.TRadiobutton', background='white')
        style.configure("White.TLabelframe", background="white")
        style.configure("White.TLabelframe.Label", background="white", foreground="#273c75",font=('Helvetica', 15)) 
        style.configure("White.TCheckbutton", background="white")
        style.configure("Blue.TButton", background="#273c75", foreground="white")
        style.configure('Grey.TLabel', background="white", foreground='grey')



        # ==============================================================================================================
        # overall layout: self  tool-main: 1-60
        # ==============================================================================================================
        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure(0, weight = 1)
        self.grid_rowconfigure(1, weight = 60)
        

        
        # ===============================================================================================================
        #  tool bar
        # ===============================================================================================================
        
        
        toolbar = tk.Frame(self, bg="#273c75")
        toolbar.grid(row = 0, column = 0, sticky = 'nsew')
        git_img = Image.open(f"{CURRENT_PATH}/github.png").resize((27,27), Image.LANCZOS)
        self.github_icon = ImageTk.PhotoImage(git_img)

        
        git_btn = tk.Button(
            toolbar,
            image=self.github_icon,
            bg="#273c75",
            command=self.open_github,
            width=27,         # Keep these if you want to enforce image size, but be aware of how Tkinter uses them
            height=27,        # These are usually in text units unless image is dominant
            padx=0,           # Set external padding to 0
            pady=0,           # Set external padding to 0
            bd=0,             # Remove border
            highlightthickness=0 # Remove highlight ring
        )
        git_btn.pack(side="left", padx=3, pady=1)
        
        doc_img = Image.open(f"{CURRENT_PATH}/document.png").resize((27,27), Image.LANCZOS)
        self.doc_icon = ImageTk.PhotoImage(doc_img)
        doc_btn = tk.Button(
            toolbar,
            image=self.doc_icon,
            bg="#273c75",
            command=self.open_document,
            width=27,         # Keep these if you want to enforce image size, but be aware of how Tkinter uses them
            height=27,        # These are usually in text units unless image is dominant
            padx=0,           # Set external padding to 0
            pady=0,           # Set external padding to 0
            bd=0,             # Remove border
            highlightthickness=0 # Remove highlight ring
        )
        doc_btn.pack(side="left", padx=3, pady=1)
        
        
        # ==============================================================================================================
        # main frame: left-right: 2-8
        # ==============================================================================================================
        
        self.main_frame = ttk.Frame(self,padding = 5, style="White.TFrame")
        self.main_frame.grid(row = 1, column = 0, sticky = 'nsew')
        
        self.main_frame.grid_rowconfigure(0, weight = 1)
        self.main_frame.grid_columnconfigure(0, weight = 2)
        self.main_frame.grid_columnconfigure(1, weight = 10) 

        # left frame
        self.left_frame = ttk.Frame(self.main_frame, padding = 5, style="White.TFrame")
        self.left_frame.grid(row = 0, column = 0, sticky = 'nsew')
        # right frame
        self.right_frame = ttk.Frame(self.main_frame,padding = 5, style="White.TFrame")
        self.right_frame.grid(row = 0, column = 1, sticky = 'nsew')





        # ==============================================================================================================
        # left: up(server_connect_frame)-bottom(bottom_param_frame): 2-8
        # ==============================================================================================================
        self.left_frame.grid_rowconfigure(0, weight = 2)
        self.left_frame.grid_rowconfigure(1, weight = 8)
        self.left_frame.grid_columnconfigure(0, weight = 1)


        # upper server connection
        self.server_connect_frame = ttk.LabelFrame(
            self.left_frame,text="Server Setup", borderwidth = 1, relief = 'groove',style="White.TLabelframe",)
        self.server_connect_frame.grid(row = 0, column = 0, sticky = 'nsew', padx=5)
        # bottom parameter setup
        self.bottom_param_frame = ttk.Frame(self.left_frame, padding = 5,style="White.TFrame")
        self.bottom_param_frame.grid(row = 1, column = 0, sticky = 'nsew')




        # ==============================================================================================================
        # 2x2 layout in parameter frame
        # ==============================================================================================================
        self.bottom_param_frame.grid_rowconfigure(0, weight = 1)
        self.bottom_param_frame.grid_rowconfigure(1, weight = 3)
        self.bottom_param_frame.grid_columnconfigure(0, weight = 1)
        self.bottom_param_frame.grid_columnconfigure(1, weight = 1)
        # upper left
        self.method_frame = ttk.LabelFrame(self.bottom_param_frame, text = 'Solver', padding = 5,style="White.TLabelframe")
        self.method_frame.grid(row = 0, column = 1, sticky = 'nsew', padx = 2, pady = 2)
        # upper right
        self.task_frame = ttk.LabelFrame(self.bottom_param_frame, text = 'Problem', padding = 5,style="White.TLabelframe")
        self.task_frame.grid(row = 0, column = 0, sticky = 'nsew', padx = 2, pady = 2)
        # lower left
        self.model_frame = ttk.LabelFrame(self.bottom_param_frame,text = 'Parameter', padding = 5,style="White.TLabelframe")
        self.model_frame.grid(row = 1, column = 0, sticky = 'nsew', padx = 2, pady = 2)
        # lower right
        self.test_param = ttk.LabelFrame(self.bottom_param_frame, text = "Test Parameter", padding = 5, style="White.TLabelframe")
        self.test_param.grid(row = 1, column = 1, sticky = 'nsew', padx = 2, pady = 2)




        # ==============================================================================================================
        # right: up-bottom 5-5
        # ==============================================================================================================
        self.right_frame.grid_rowconfigure(0, weight = 1)
        self.right_frame.grid_rowconfigure(1, weight = 5)
        self.right_frame.grid_columnconfigure(0, weight = 1)


        self.console_label_frame = ttk.LabelFrame(self.right_frame, text="Log File Output",style="White.TLabelframe")
        self.console_label_frame.grid(row = 0, column = 0, sticky = 'nsew')
        
        self.console_label_frame.grid_rowconfigure(0, weight = 1)
        self.console_label_frame.grid_columnconfigure(0, weight = 1)
        
        self.console_output_frame = ttk.Frame(self.console_label_frame,style="White.TFrame")
        self.console_output_frame.grid(row = 0,column=0, sticky = 'nsew')

          
        self.summary_frame = ttk.LabelFrame(self.right_frame,padding=5, text="Summary Output",style="White.TLabelframe")
        self.summary_frame.grid(row = 1, column = 0, pady=6, sticky = 'nsew',)
        
        self.summary_frame.grid_rowconfigure(0, weight = 1)
        self.summary_frame.grid_columnconfigure(0, weight = 1)
        
        self.chart_output_frame = ttk.Frame(self.summary_frame,style="White.TFrame")
        self.chart_output_frame.grid(row = 0, column = 0, sticky = 'nsew')
        
        self.summary_frame.grid_rowconfigure(0, weight = 1)
        self.summary_frame.grid_columnconfigure(0, weight = 1)
        


        # ==============================================================================================================
        # console output area (Log File Output)
        # ==============================================================================================================
        self.console_output_frame.grid_rowconfigure(0, weight = 1)
        self.console_output_frame.grid_columnconfigure(0, weight = 1)
        self.log_area = scrolledtext.ScrolledText(
            self.console_output_frame,
            wrap = tk.WORD,
            state = 'disabled',
            bg = 'white',
            fg = 'black',
            insertbackground = 'black',
            font = ('Consolas', 10),
            relief="flat",
            )
        self.log_area.grid(row = 0,column=0, sticky = 'nsew', padx = 5, pady = 3)
        # self.log_area.pack(padx = 5, pady = 5, fill = tk.BOTH, expand = True)
        
        
        
        # initial insert text
        text = "#############################################\n#                                           #\n#    W E L C O M E   T O   E A S Y N C O !  #\n#                                           #\n#############################################"
        self.log_area.config(state = 'normal')
        
        

        for text in TEXT:
            self.log_area.insert(tk.END, text)
        
        self.log_area.config(state = 'disabled')


        # ==============================================================================================================
        # summary output area (Summary Output)
        # ==============================================================================================================
        self.summary_canvas = tk.Canvas(self.chart_output_frame, bg="white")
        # self.canvas.grid(row = 0, column = 0, sticky = 'nsew')
        self.summary_canvas.pack(padx = 5, pady = 5, fill = tk.BOTH, expand = True)
        self.summary_scroll_bar = ttk.Scrollbar(self.chart_output_frame, orient = 'vertical', command = self.summary_canvas.yview)

        image = Image.open(f"{CURRENT_PATH}/logo2.png") 
        image.thumbnail((350, 350))
        self.photo = ImageTk.PhotoImage(image)
        
        
        self.summary_canvas.bind("<Configure>", self.draw_centered_image)
        
        self.summary_canvas.create_image(
        self.summary_canvas.winfo_width() / 2,
         self.summary_canvas.winfo_height() / 2,
        image=self.photo,
        anchor=tk.CENTER,
        tags="image")



        # ==============================================================================================================
        # Add other widgets
        # ==============================================================================================================
        # SSH related settings
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        self.connected = False
        self.trainsport = None
        self.is_running = False
        self.last_position = 0
        self.log_path = None
        
        
        
        
        # ==============================================================================================================
        # Connecting to remote server 
        # ==============================================================================================================
        self.server_connect_frame.columnconfigure(1, weight = 1)
        ttk.LabelFrame(self.server_connect_frame,
                text = 'Server setups',
                style="White.TLabelframe",
                padding= 5,
                  ).grid(
            row = 0, column = 0, columnspan = 2, sticky = 'w', padx = 5, pady = 5
        )
                  
        tk.Label(self.server_connect_frame, text = 'host:',bg = "white",).grid(row = 1, column = 0, sticky = 'e', padx = 5, pady = 3)
        self.host_var = tk.StringVar(value = HOSTNAME)
        ttk.Entry(self.server_connect_frame, textvariable=self.host_var,).grid(row = 1, column = 1, sticky = 'ew', padx = 5, pady = 3)

        tk.Label(self.server_connect_frame, text = 'port:',bg = "white").grid(row = 2, column = 0, sticky = 'e', padx = 5, pady = 3)
        self.port_var = tk.IntVar(value = PORT)
        ttk.Entry(self.server_connect_frame, textvariable=self.port_var, ).grid(row = 2, column = 1, sticky = 'ew', padx = 5, pady = 3)
        
        tk.Label(self.server_connect_frame, text='username:',bg = "white").grid(row=3, column=0, sticky='e', padx=5, pady=3)
        self.username_var = tk.StringVar(value=USERNAME)
        ttk.Entry(self.server_connect_frame, textvariable=self.username_var, ).grid(row=3, column=1, sticky='ew', padx=5, pady=3)

        
        self.login_method = tk.StringVar(value="key")  # 默认用key
        
        #self.login_method_frame = self.server_connect_frame.grid(row=3, column=0, sticky='e', padx=5, pady=3)
        login_method_frame = ttk.Frame(self.server_connect_frame,style="White.TFrame")
        login_method_frame.grid(row=4, column=1, columnspan=2, sticky='w', padx=5, pady=3)

        tk.Label(self.server_connect_frame, text='login method:',bg = "white").grid(row=4, column=0, sticky='e', padx=5, pady=3)
        ttk.Radiobutton(login_method_frame, text='key', variable=self.login_method, value='key', command=self.update_login_method,style='Custom.TRadiobutton').grid(row=4, column=1, sticky='w')
        ttk.Radiobutton(login_method_frame, text='password', variable=self.login_method, value='password', command=self.update_login_method,style='Custom.TRadiobutton').grid(row=4, column=2, sticky='w')
        
        # 密钥输入
        self.key_var = tk.StringVar(value=KEY)
        self.key_entry = ttk.Entry(self.server_connect_frame, textvariable=self.key_var, width=1)
        self.key_label = tk.Label(self.server_connect_frame, text='key:',bg = "white")

        # 密码输入
        self.password_var = tk.StringVar(value=PASSWORD)
        self.password_entry = ttk.Entry(self.server_connect_frame, textvariable=self.password_var, width=1, show='*')
        self.password_label = tk.Label(self.server_connect_frame, text='password:',bg = "white")

        # 初始只显示密钥输入
        self.key_label.grid(row=5, column=0, sticky='e', padx=5, pady=3)
        self.key_entry.grid(row=5, column=1, sticky='ew', padx=5, pady=3)
        
        
        tk.Label(self.server_connect_frame, text='device:',bg = "white").grid(row=6, column=0, sticky='e', padx=5, pady=3)
        self.device_var = tk.StringVar(value=DEVICE)
        ttk.Entry(self.server_connect_frame, textvariable=self.device_var, width = 1).grid(row=6, column=1, sticky='ew', padx=5, pady=3)
        

        # Confirm button
        style.configure("Custom.TButton", background="white")
        self.connect_btn = tk.Button(self.server_connect_frame, text = 'Connect',
                   command = self.on_connect_click, bg="#273c75",foreground='white',font=('Segoe UI', 10, 'bold'),height=1)
        self.connect_btn.grid(row = 7, column = 1, sticky = 'ew',padx=5, pady=3)





        # ==============================================================================================================
        # method display
        # ==============================================================================================================
        self.method_frame.columnconfigure(0, weight = 1)
        #self.method_frame.columnconfigure(1, weight = 7)
        self.method_frame.rowconfigure(0, weight = 8)
        
        self.method_frame.rowconfigure(1, weight = 2)

        List_frame = ttk.Frame(self.method_frame, style="White.TFrame")
        # ttk.Frame(self.task_frame,style="White.TFrame")
        List_frame.grid(row = 0, column = 0, sticky = 'nsew')
        
        methods = METHOD_LIST

        scrollBar_method = tk.Scrollbar(List_frame)
        scrollBar_method.pack(side = 'right', fill = 'y')
        self.methodContent = tk.Listbox(List_frame, yscrollcommand = scrollBar_method.set, width =5,exportselection=False)
        # taskContent = tk.Listbox(task_list_frame, yscrollcommand = scrollBar.set,height=5, width = 1)
        for i in range(len(methods)):
            self.methodContent.insert('end', methods[i])
        self.methodContent.pack(side = 'left', fill = 'both', expand = True)
        self.methodContent.bind('<ButtonRelease-1>', lambda event: self.on_select_method_click(self.methodContent))
        
        self.methodContent.bind('<FocusIn>', lambda e: self.methodContent.config(relief='sunken'))
        self.methodContent.bind('<FocusOut>', lambda e: self.methodContent.config(relief='flat'))

        scrollBar_method.config(command = self.methodContent.yview)

        self.figure_frame = ttk.Frame(self.method_frame, style="White.TFrame")
        self.figure_frame.grid(row = 1, column = 0, sticky = 'nsew')
        
        self.figure_frame.columnconfigure(0, weight = 1)
        self.figure_frame.columnconfigure(1, weight = 1)
        
        self.save_btn = tk.Button(self.figure_frame, text = 'Select', width=5, font=('Segoe UI', 10, 'bold'), command = self.save_click, bg='#273c75',foreground='white', height=1)
        self.save_btn.grid(row = 0, column = 0, sticky = 'ew',padx=5, pady=3)
        
        # stop buttom
        self.plot_btn = tk.Button(self.figure_frame, text = 'Plot', width=5, font=('Segoe UI', 10, 'bold'),command = self.plot_click,bg='#273c75',foreground='white',height=1)
        self.plot_btn.grid(row = 0, column = 1, sticky = 'ew',padx=5, pady=3)


        # ==============================================================================================================
        # Model Param settings
        # ==============================================================================================================
        self.model_frame.columnconfigure(0, weight = 1)
        self.model_frame.columnconfigure(1, weight = 0)
        self.model_frame.rowconfigure(0,weight=1)

        
        self.param_canvas = tk.Canvas(self.model_frame,background="white",width=180) # Canvas can have its own background
        self.param_canvas.grid(row=0, column=0, sticky="nsew")
        # self.param_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        
        # 垂直滚动条
        self.param_scrollbar = ttk.Scrollbar(self.model_frame, orient="vertical", command=self.param_canvas.yview)
        self.param_scrollbar.grid(row=0, column=1, sticky="ns")
        self.param_canvas.configure(yscrollcommand=self.param_scrollbar.set)
        
        # 水平滚动条
        self.param_scrollbar_x = ttk.Scrollbar(self.model_frame, orient="horizontal", command=self.param_canvas.xview)
        self.param_scrollbar_x.grid(row=1, column=0, sticky="ew")
        self.param_canvas.configure(xscrollcommand=self.param_scrollbar_x.set)
        
        self.param_canvas.grid_rowconfigure(0,weight=1)
        self.param_canvas.grid_columnconfigure(0,weight=1)
        
        self.param_scrollable_frame = ttk.Frame(self.param_canvas, style="White.TLabelframe") # Apply style if desired
        self.param_scrollable_frame.grid(row=0,column=0, sticky="nsew")
        self.param_canvas_frame_id = self.param_canvas.create_window((0, 0), window=self.param_scrollable_frame, anchor="nw")
        
        self.param_scrollable_frame.grid_columnconfigure(0, weight=1)
        # self.param_scrollable_frame.grid_rowconfigure(0, weight=1)
        # self.param_scrollable_frame.grid_columnconfigure(1, weight=1)
        
        
        self.param_scrollable_frame.bind("<Configure>", self.on_frame_configure)
        self.param_canvas.bind("<Configure>", self.on_canvas_configure)
        


        
        # ==============================================================================================================
        # task display
        # ==============================================================================================================

        self.task_frame.rowconfigure(0, weight = 8)
        self.task_frame.rowconfigure(1, weight = 2)
        self.task_frame.columnconfigure(0, weight = 1)


        task_list_frame = ttk.Frame(self.task_frame,style="White.TFrame")
        task_list_frame.grid(row = 0, column = 0, sticky = 'nsew')
        self.task_param_frame = ttk.Frame(self.task_frame,style="White.TFrame")
        self.task_param_frame.grid(row = 1, column = 0, sticky = 'nsew')

        taskList = TASK_LIST
        scrollBar = tk.Scrollbar(task_list_frame)
        scrollBar.pack(side = 'right', fill = 'y')
        taskContent = tk.Listbox(task_list_frame, yscrollcommand = scrollBar.set,height=5, width = 1,exportselection=False)
        for i in range(len(taskList)):
            taskContent.insert('end', taskList[i])
        taskContent.pack(side = 'left', fill = 'both', expand = True)
        taskContent.bind('<ButtonRelease-1>', lambda event: self.on_select_task_click(taskContent))
        # taskContent.bind('<<ListboxSelect>>', self.on_select_task_click)
        taskContent.bind('<FocusIn>', lambda e: taskContent.config(relief='sunken'))
        taskContent.bind('<FocusOut>', lambda e: taskContent.config(relief='flat'))
        scrollBar.config(command = taskContent.yview)
        #self.bind('<Button-1>', lambda event: self.on_click_outside(event, taskContent))

        self.task_param_frame.grid_columnconfigure(0, weight = 0)
        self.task_param_frame.grid_columnconfigure(1, weight = 10)
        self.task_param_frame.grid_columnconfigure(2, weight = 10)
        self.task_param_frame.grid_columnconfigure(3, weight = 0)
        self.task_param_frame.grid_columnconfigure(4, weight = 10)



        self.scale_check_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.task_param_frame, variable=self.scale_check_var,style='White.TCheckbutton').grid(row=0, column=0, sticky='w')

        # 2. 将此行其余控件的 column 索引 +1
        tk.Label(self.task_param_frame, text='scale', bg="#273c75", foreground="white", width=8, font=('Segoe UI', 10, 'bold')).grid(row=0, column=1, sticky='nsew', padx=2, pady=3)
        self.problem_size_var = tk.IntVar(value=100)
        self.problem_size_max_var = tk.IntVar(value=500)
        ttk.Entry(self.task_param_frame, textvariable=self.problem_size_var, width=5).grid(row=0, column=2, sticky='ew', padx=5, pady=3)
        self.scale_line = tk.Label(self.task_param_frame, text='—', bg="white", foreground="#273c75", width=1)
        self.scale_line.grid(row=0, column=3, sticky='nsew')
        ttk.Entry(self.task_param_frame, textvariable=self.problem_size_max_var, width=5).grid(row=0, column=4, sticky='ew', padx=5, pady=3)

        # --- capacity range ---
        # 1. 为 "capacity" 行添加布尔变量和复选框
        self.capacity_check_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.task_param_frame, variable=self.capacity_check_var,style='White.TCheckbutton').grid(row=1, column=0, sticky='w')

        # 2. 将此行其余控件的 column 索引 +1
        tk.Label(self.task_param_frame, text='capacity', bg="#273c75", foreground="white", width=8, font=('Segoe UI', 10, 'bold')).grid(row=1, column=1, sticky='nsew', padx=2, pady=3)
        self.capacity_var = tk.IntVar()
        self.capacity_max_var = tk.IntVar()
        ttk.Entry(self.task_param_frame, textvariable=self.capacity_var, width=5).grid(row=1, column=2, sticky='ew', padx=5, pady=3)
        self.scale_line_cap = tk.Label(self.task_param_frame, text='—', bg="white", foreground="#273c75", width=1)
        self.scale_line_cap.grid(row=1, column=3, sticky='nsew')
        ttk.Entry(self.task_param_frame, textvariable=self.capacity_max_var, width=5).grid(row=1, column=4, sticky='ew', padx=5, pady=3)

        # --- distribution_list ---
        # 1. 为 "distribution" 行添加布尔变量和复选框
        """"self.distribution_check_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.task_param_frame, variable=self.distribution_check_var,style='White.TCheckbutton').grid(row=2, column=0, sticky='w')"""
        
        # 2. 将此行其余控件的 column 索引 +1
        """tk.Label(self.task_param_frame, text='distribution', bg="#273c75", foreground="white", width=8, font=('Segoe UI', 10, 'bold')).grid(row=2, column=1, sticky='nsew', padx=2, pady=3)
        self.distribution_var1 = tk.StringVar(value='uniform')
        self.distribution_var2 = tk.StringVar(value='cluster')
        ttk.Entry(self.task_param_frame, textvariable=self.distribution_var1, width=5).grid(row=2, column=2, sticky='ew', padx=5, pady=3)
        self.scale_line_dis = tk.Label(self.task_param_frame, text='—', bg="white", foreground="#273c75", width=1)
        self.scale_line_dis.grid(row=2, column=3, sticky='nsew')
        ttk.Entry(self.task_param_frame, textvariable=self.distribution_var2, width=5).grid(row=2, column=4, sticky='ew', padx=5, pady=3)"""

        # ==============================================================================================================
        # test setting
        # ==============================================================================================================

        # self.test_param.columnconfigure(1, weight = 1)
        
        self.test_param.grid_columnconfigure(0, weight = 1)
        self.test_param.grid_columnconfigure(1, weight = 2)
        self.test_param.grid_rowconfigure(0, weight = 6)
        self.test_param.grid_rowconfigure(1, weight = 1)
        
        self.test_entry_frame = ttk.Frame(self.test_param, style="White.TFrame")
        self.test_entry_frame.grid(row = 0, column = 0, sticky = 'nsew')
        # self.test_entry_frame.pack(padx=3, fill="x", expand=False) 
        
        # Method
        self.select_model = tk.StringVar(value=None)
        ttk.Label(self.test_entry_frame, text = 'Method',style="White.TLabel").grid(row = 0, column = 0, sticky = 'e', padx =5, pady = 3)
        self.method_var = tk.StringVar(value = None)
        #self.select_model.trace_add('write', self.update_method_label)
        self.method_label = ttk.Label(self.test_entry_frame, text = None, style="White.TLabel")
        self.method_label.grid(row = 0, column = 1, sticky = 'ew', padx = 5, pady = 3)
        #ttk.Entry(self.test_param, textvariable = self.metohd_var).grid(row = 2, column = 1, sticky = 'ew', padx = 5, pady = 3)
        
        
        # Task
        ttk.Label(self.test_entry_frame, text = 'Task',style="White.TLabel").grid(row = 1, column = 0, sticky = 'e', padx =5, pady = 3)
        self.task_var = tk.StringVar(value = None)
        #self.task_var.trace('w', self.select_task)
        self.task_label = ttk.Label(self.test_entry_frame, text = None,style="White.TLabel")
        self.task_label.grid(row = 1, column = 1, sticky = 'ew', padx = 5, pady = 3)

        
        # mode settings
        MODE_LIST = ['test', 'train']
        ttk.Label(self.test_entry_frame, text = 'mode',style="White.TLabel").grid(row = 2, column = 0, sticky = 'e', padx = 5, pady = 3)
        self.mode_var = tk.StringVar(value = 'test')
        self.select_mode = 'test'
        mode = ttk.Combobox(self.test_entry_frame, textvariable = self.mode_var, values = MODE_LIST, state = 'readonly',width = 5)
        mode.grid(row = 2, column = 1, sticky = 'ew', padx = 5, pady = 3)
        mode.bind('<<ComboboxSelected>>', lambda event: self.on_select_mode_click(mode))
        

        # Batch size
        ttk.Label(self.test_entry_frame, text = 'Batch_size',style="White.TLabel").grid(row = 3, column = 0, sticky = 'e', padx =5, pady = 3)
        self.batch_var = tk.IntVar(value = 128)
        ttk.Entry(self.test_entry_frame, textvariable = self.batch_var, width=15).grid(row = 3, column = 1, sticky = 'ew', padx = 5, pady = 3)
        
        # Episode
        ttk.Label(self.test_entry_frame, text = 'Episode',style="White.TLabel").grid(row = 4, column = 0, sticky = 'e', padx =5, pady = 3)
        self.episode_var = tk.IntVar(value = 1280)
        ttk.Entry(self.test_entry_frame, textvariable = self.episode_var, width=15).grid(row = 4, column = 1, sticky = 'ew', padx = 5, pady = 3)

        # Decoder_strategy
        self.decoder_strategy_label = ttk.Label(self.test_entry_frame, text = 'Decoder\nStrategy',style="White.TLabel")
        self.decoder_strategy_label.grid(row = 5, column = 0, sticky = 'e', padx =5, pady = 3)
        self.decoder_strategy_var = tk.StringVar(value = 'greedy')
        self.decoder_strategy_entry = ttk.Entry(self.test_entry_frame, textvariable = self.decoder_strategy_var, width=15)
        self.decoder_strategy_entry.grid(row = 5, column = 1, sticky = 'ew', padx = 5, pady = 3)
        
        #===============================================================================================================
        # dataset settings
        #===============================================================================================================

        # val_dataset
        self.dataset_options = ['dataset1', 'dataset2','dataset3']
        # ttk.Label(self.test_param, text = 'Val_Datasets').grid(row = 5, column = 0, sticky = 'e', padx =5, pady = 3)
        self.dataset_btn = tk.Button(self.test_entry_frame, text = 'Datasets', command = self.on_select_dataset_click,
                                     foreground="#273c75",
                                      bg = "white",
                                      width = 1,
                                     )
        self.dataset_btn.grid(row = 6, column = 0, sticky = 'nesw', padx =5, pady = 3)
        defualt_dataset =  DATASET
        
        self.val_datasets_var = tk.StringVar(value = defualt_dataset)
    
    
        self.dataset = ttk.Entry(self.test_entry_frame, textvariable = self.val_datasets_var, state='normal',style="White.TEntry", width=15)
        self.dataset.grid(row = 6, column = 1, sticky = 'ew', padx = 5, pady = 3)
        
        self.dataset.bind("<Enter>", lambda event:self.on_enter(self.dataset))
        self.dataset.bind("<Leave>", lambda event:self.on_leave(self.dataset))
        
        
        
        # checkpoint_file

        self.ckpt_btn = tk.Button(self.test_entry_frame, text = 'Checkpoint', command = self.on_select_ckpt_click,
                                  foreground='#273c75',bg='white',)
        self.ckpt_btn.grid(row = 7, column = 0, sticky = 'nesw', padx =5, pady = 3)
        self.val_ckpt_var = tk.StringVar(value = "<Choose Path>") 
        defalut_ckpt = CKPT
        # defalut_ckpt = '/public/home/luoqing/code/EasyNCO_v0621/EasyNCO/results/train/pomo_tsp/2025-06-21-11-03-44_pomo_tsp_100_train/checkpoints/pomo_tsp100_epoch04.ckpt'
        self.val_ckpt_var = tk.StringVar(value = defalut_ckpt)
        

        self.method_flag = None
        self.task_flag = None
        self.ckpt = ttk.Entry(self.test_entry_frame, textvariable = self.val_ckpt_var, state='normal',style="White.TEntry", width=15)
        self.ckpt.grid(row = 7, column = 1, sticky = 'ew', padx = 5, pady = 3)
        self.ckpt.bind("<Enter>", lambda event:self.on_enter(self.ckpt))
        self.ckpt.bind("<Leave>", lambda event:self.on_leave(self.ckpt))
        
        
        #============================================================================================
        # start buttom
        self.bottom_frame = ttk.Frame(self.test_param, style="White.TFrame")
        self.bottom_frame.grid(row=1, column=0, sticky='nsew')
        # self.bottom_frame.pack(padx=5, fill="x")
        self.bottom_frame.grid_columnconfigure(0, weight = 1)
        self.bottom_frame.grid_columnconfigure(1, weight = 1)
        self.bottom_frame.grid_rowconfigure(0, weight = 1)
        
        self.start_btn = tk.Button(self.bottom_frame, text = 'Start',font=('Segoe UI', 10, 'bold'), command = self.start_click, bg='#273c75',foreground='white', height=1)
        self.start_btn.grid(row = 0, column = 0, sticky = 'ew',padx=5, pady=3)
        
        # stop buttom
        self.stop_btn = tk.Button(self.bottom_frame, text = 'Stop', font=('Segoe UI', 10, 'bold'),command = self.stop_click,bg='#273c75',foreground='white',height=1)
        self.stop_btn.grid(row = 0, column = 1, sticky = 'ew',padx=5, pady=3)
        



    # MainApp.on_connect_click
    def on_connect_click(self):
        
        if not self.connected:
            HOSTNAME = self.host_var.get()
            PORT = self.port_var.get()
            KEY = self.key_var.get()
            PASSWORD = self.password_var.get()
            USERNAME = self.username_var.get()
    
                
            if self.login_method.get() == 'key':
                try:
                    warnings.filterwarnings("ignore")
                    self.client.connect(
                    hostname=HOSTNAME,
                    port=PORT,
                    username=USERNAME,
                    key_filename=KEY,
                    look_for_keys=False,
                    allow_agent=False,
                    timeout=10)
                except Exception as e:
                    print(f"connection fail: {e}")
                    self.log_area.config(state = 'normal')
                    self.log_area.insert(tk.END, f"Connection Error: {str(e)}\n")
                    self.log_area.config(state = 'disabled')
                    
                            
            if self.login_method.get() == 'password':
                print("password")
                try:
                    warnings.filterwarnings("ignore")
                    self.client.connect(
                    hostname=HOSTNAME,
                    port=PORT,
                    username=USERNAME,
                    password=PASSWORD,
                    look_for_keys=False,
                    allow_agent=False,
                    timeout=10)
                except Exception as e:
                    print(f"connection fail: {e}")
                    self.log_area.config(state = 'normal')
                    self.log_area.insert(tk.END, f"Connection Error: {str(e)}\n")
                    self.log_area.config(state = 'disabled')
                    

            self.connect_btn.config(text = 'Disconnect')
                
            self.log_area.config(state = 'normal')
            self.log_area.delete('1.0', tk.END) 
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_area.insert(tk.END, f"[{current_time}] Connection Success: Connect to {HOSTNAME}.\n")
            self.log_area.config(state = 'disabled')
                
            self.sftp = self.client.open_sftp()
            self.transport = self.client.get_transport()
                
            self.connected = True
            
                
        else:
            self.stop_monitoring()   


            
            


    # MainApp.stop_monitoring
    def stop_monitoring(self):
        self.is_running = False
        self.connect_btn.config(text = 'Connect')
        self.connected = False
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_area.config(state = 'normal')
        self.log_area.insert(tk.END, f"[{current_time}] Connection closed.\n")
        self.log_area.config(state = 'disabled')
        
        self.status_var.set("ON")
        
        if self.client:
            self.client.close()
            
            # inintialize the client
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
            # self.client = None


    # MainApp.on_select_method_click
    def on_select_method_click(self, methodContent):
        
        if not hasattr(self, 'method_list'):
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.update_log_display(f"[{current_time}]  [Warnings] Please select a task first.")
            
        else:
        
            select_index = methodContent.curselection()
            if select_index:
                self.select_model = methodContent.get(select_index)
                self.method_var.set(self.select_model)
                self.method_var.trace_add("write", self.update_method_label(self.select_model))
            
            methodContent.configure(selectbackground='#ffffc4', selectforeground='black')



    def on_click_outside(self, event, taskContent):
        # 判断鼠标点击是否在 taskContent 区域之外
        x1 = taskContent.winfo_x()
        y1 = taskContent.winfo_y()
        x2 = x1 + taskContent.winfo_width()
        y2 = y1 + taskContent.winfo_height()
        if (x1 <= event.x <= x2 and y1 <= event.y <= y2):
            print("Clicked outside of taskContent")  
            if self.select_task:
                taskContent.configure(selectbackground='#ffffc4', selectforeground='black')  # 选中状态
            else:
                taskContent.config(relief='flat', bg='lightgrey')    # 默认状态

        

        

    # MainApp.on_select_task_click
    def on_select_task_click(self, taskContent):

        self.select_task = taskContent.get(taskContent.curselection())
        
        self.task_var.set(self.select_task)
        self.task_var.trace_add("write", self.update_task_label(self.select_task))
        
        taskContent.configure(selectbackground='#ffffc4', selectforeground='black')
        

          
        if self.select_task == 'jssp' or self.select_task == 'fjsp': 

            self.scale_line.config(text="×")
                
        else:
            self.scale_line.config(text='—')
         
        # update methodcontent
        self.method_list = return_task_method_list(self.select_task, UTILS_YAML)
        self.update_method_listbox(method_list = self.method_list)   
        
        print(f'task {self.select_task} has been selected')
        
        return None

    def start_click(self):
        
        self.print_watermark()
        
        
        self.summary_flag = True
        self.summary_line = []
        self.score = []
        self.loss = []
        self.eval_score = []
        
        

        
        if self.select_mode == 'test':
        # self.output_summary()
            self.update_summary_test()
            
            """self.min_scale = self.problem_size_var.get() if self.scale_check_var.get() else None
            self.max_scale = self.problem_size_max_var.get() if self.scale_check_var.get() else None
            
            self.min_capacity = self.capacity_var.get() if self.capacity_check_var.get() else None
            self.max_capacity = self.capacity_max_var.get() if self.capacity_check_var.get() else None
            
            self.distribution_1 = self.distribution_var1.get() if self.distribution_check_var.get() else None
            self.distribution_2 = self.distribution_var2.get() if self.distribution_check_var.get() else None"""
            
        
            run_cfg = {'test_loader': {'model_filename': None},
                       'model_parameters': {},
                       'varying_data_params':{'scale_range':None,
                                              'capacity_range': None,
                                              'distribution_list': None,},
            }

            run_cfg['mode'] = self.select_mode
            run_cfg['model'] =self.select_model
            run_cfg['problem'] = self.select_task
            
            """if self.min_scale == self.max_scale:
                run_cfg['scale'] = self.min_scale
                self.scale = self.problem_size_var.get()
                
                varying_data_params_set = [f"varying_data_params.{k}=null"   for k, v in run_cfg['varying_data_params'].items()]
                varying_data_params_set = " ".join(varying_data_params_set)
            else:
                run_cfg['scale'] = self.min_scale
                run_cfg['varying_data_params']['scale_range'] = [self.min_scale,self.max_scale]
                #run_cfg['varying_data_params']['capacity_range'] = [self.min_capacity,self.max_capacity]
                
                varying_data_params_set = [f"varying_data_params.{k}='{v}'"   for k, v in run_cfg['varying_data_params'].items()]
                varying_data_params_set = " ".join(varying_data_params_set)"""
                
                
            if self.scale_check_var.get():
                run_cfg['varying_data_params']['scale_range'] = [self.problem_size_var.get(),self.problem_size_max_var.get()]

            if self.capacity_check_var.get():
                run_cfg['varying_data_params']['capacity_range'] = [self.capacity_var.get(),self.capacity_max_var.get()]
                
            """if self.distribution_check_var.get():
                run_cfg['varying_data_params']['distribution_list'] = [self.distribution_var1.get(),self.distribution_var2.get()]"""
               
               
            varying_data_params_set =[] 
            for k, v in run_cfg['varying_data_params'].items():
                if v is None or v == []:
                    varying_data_params_set.append(f"varying_data_params.{k}=null")
                    continue
                
                if self.select_task == 'jssp' or self.select_task == 'fjsp':   
                    varying_data_params_set.append(f"num_job={v[0]}")
                    varying_data_params_set.append(f"num_machine={v[1]}")
                    break
                
                varying_data_params_set.append(f"varying_data_params.{k}='{v}'")
                
            varying_data_params_set = " ".join(varying_data_params_set)

                
                
            """if self.min_scale == self.max_scale:
                    run_cfg['scale'] = self.min_scale
                    varying_data_params_set = [f"varying_data_params.{k}=null"   for k, v in run_cfg['varying_data_params'].items()]
                    varying_data_params_set = " ".join(varying_data_params_set)
                else:
                    run_cfg['scale'] = self.min_scale
                    run_cfg['varying_data_params']['scale_range'] = [self.min_scale,self.max_scale]
                    
                    varying_data_params_set = [f"varying_data_params.{k}='{v}'"   for k, v in run_cfg['varying_data_params'].items()]
                    varying_data_params_set = " ".join(varying_data_params_set)"""
                
                
            # module:batch_size
            run_cfg['batch_size'] = self.batch_var.get()
            # module:episodes
            run_cfg['episodes'] = self.episode_var.get()
            self.episodes = self.episode_var.get()
            # module:decoder_strategy
            run_cfg['decoder_strategy'] = self.decoder_strategy_var.get()
            self.decoder_strategy = self.decoder_strategy_var.get()
            # module:val_data_path
            # test_cfg['test_data_path'] = self.val_datasets_var.get()
            # test_loader:model_dirpath
            full_ckpt_path = self.val_ckpt_var.get()
            ckpt_file = full_ckpt_path.split('/')[-1]
            ckpt_dir = full_ckpt_path.replace(ckpt_file, '')
            run_cfg['test_loader']['model_filename'] = ckpt_file
            run_cfg['test_loader']['model_dirpath'] = ckpt_dir
            
            # config: test_data_path
            full_data_path = self.val_datasets_var.get()
            separator = "datasets/"
            # 检查分隔符是否存在
            if separator in full_data_path:
            # 分割字符串，并取[1]索引（即分隔符之后的部分）
                run_cfg["test_data_path"] = full_data_path.split(separator)[1]
            else:
                self.insert_text_in_log("Please put the test_data in datasets directory!")
            # run_cfg["test_data_path"] = full_data_path.split('/')[-2] + '/' + full_data_path.split('/')[-1] 
        
        
            cmd = 'nohup'
            log_route = GUI_DIR + '/GUI_log.txt'
            # log_route = 'GUI/GUI_log.txt'
            exe_file = "eval.py"
        
            
            """for k ,v in self.additional_vars.items():             
                if v == "True" or v == "False":
                    run_cfg['model_parameters'][k] = True if v == "True" else False      
                else:
                    run_cfg['model_parameters'][k] = v"""
                    
            for label_text, var, original_type in self.model_param_vars:
                v = var.get()
                
                if v == "None":
                    
                    print("label_text: value is None: ", label_text)
                
                
                
                """if v == "":
                    run_cfg['model_parameters'][label_text] = None"""
                
                if original_type is list:
                    if not v:
                        final_value = []

                    else:
                        items_as_strings = [item.strip() for item in v.split(',')]
                        processed_items = []
                        for item in items_as_strings:
                            try:
                                processed_items.append(int(item))
                            except ValueError:
                                try:
                                    processed_items.append(float(item))
                                except ValueError:
                                    processed_items.append(item)
                        run_cfg['model_parameters'][label_text] =f"'{processed_items}'"
                
                elif original_type is bool:
                    run_cfg['model_parameters'][label_text] = str(v).lower() == 'true'
                    
                else:
                    
                    run_cfg['model_parameters'][label_text] = v
               
                
            model_param_set = []   
            for k,v in run_cfg['model_parameters'].items():
                if v == "":
                    model_param_set.append(f"settings.{k}=null")  
                    continue                 
                model_param_set.append(f"settings.{k}={v}")

                    
            # model_param_set = [f"settings.{k}={v}"   for k, v in run_cfg['model_parameters'].items()]
            model_param_set = " ".join(model_param_set)
            
            
            params_set = f" settings={run_cfg['model']}_settings" + \
            f" mode={run_cfg['mode']}" + \
            f" model={run_cfg['model']}" + \
            f" problem={run_cfg['problem']}" + \
            f" cuda=[{self.device_var.get()[-1]}]" + \
            f" batch_size={run_cfg['batch_size']}" + \
            f" episodes={run_cfg['episodes']}" + \
            f" decoder_strategy={run_cfg['decoder_strategy']}" + \
            f" test_data_path={run_cfg['test_data_path']}" + \
            f" settings.test_loader.model_filename={run_cfg['test_loader']['model_filename']}"   
            
            #   f" test_data_path={test_cfg['test_data_path']}" + \
            # f" scale={run_cfg['scale']}" + \
        
            exe_command = (
                f"cd {ROOT_DIR} && "
                f"({cmd} "
                f"{PYTHON_ROUTE} "
                f"-u "
                f"{exe_file} "
                f" {params_set}"
                f" {model_param_set} "
                f" {varying_data_params_set} "
                f"> {log_route} 2>&1 "
                "& echo $!)")
                
            """exe_command = (
                f"(cd {ROOT_DIR}; " # 在子shell内部，用分号也可以
                f"exec {cmd} "
                f" {PYTHON_ROUTE} {exe_file} {params_set} {model_param_set} "
                f"> {log_route} 2>&1"
                f") & echo $!")"""
            
         
            print(exe_command)
            # exe_command = cmd + ' ' + PYTHON_ROUTE + ' ' + ROOT_DIR + '/' + exe_file + f' --cfg_name {yaml_path} ' + ' >' + log_route + ' 2>&1 &'
            self.log_path = log_route
            print('Running ')
            command = exe_command
            
            stdin, stdout, stderr = self.client.exec_command(command)
            self.pid = stdout.readline().strip()  # 这里会直接得到PID
            print(f"PID: {self.pid}")
            
            self.is_running = True
            threading.Thread(
                target=self.monitor_log,
                args=(self.log_path,),
                daemon=True
            ).start() ##!!!
            
            

    
            
        elif self.select_mode == 'train':
            
            
            self.update_summary_test()
        
            run_cfg = {'model_parameters': {},}
        
            run_cfg['mode'] = self.select_mode
            run_cfg['model'] =self.select_model
            run_cfg['problem'] = self.select_task
            run_cfg['scale'] = self.problem_size_var.get()
            self.scale = self.problem_size_var.get()
            # module:batch_size
            run_cfg['batch_size'] = self.batch_var.get()
            self.batch_size = self.batch_var.get()
            # module:episodes
            run_cfg['episodes'] = self.episode_var.get()
            self.episodes = self.episode_var.get()
            run_cfg['max_epochs'] = self.epoch_var.get()
            # module:val_data_path
            # test_cfg['test_data_path'] = self.val_datasets_var.get()
            # test_loader:model_dirpath
            """full_ckpt_path = self.val_ckpt_var.get()
            ckpt_file = full_ckpt_path.split('/')[-1]
            ckpt_dir = full_ckpt_path.replace(ckpt_file, '')
            run_cfg['test_loader']['model_filename'] = ckpt_file
            run_cfg['test_loader']['model_dirpath'] = ckpt_dir"""
            
            full_ckpt_path = self.val_ckpt_var.get()
            run_cfg['ckpt_path'] = full_ckpt_path
            
            # config: test_data_path
            full_data_path = self.val_datasets_var.get()
            run_cfg["val_data_path"] = full_data_path.split('/')[-2] + '/' + full_data_path.split('/')[-1] 
        
            # model_parameters
            # run_cfg['model_parameters'] = self.additional_vars
            """for k ,v in self.additional_vars.items():             
                if v == "True" or v == "False":
                    run_cfg['model_parameters'][k] = True if v == "True" else False      
                else:
                    run_cfg['model_parameters'][k] = v"""
                    
            model_param_set = [f"settings.model.{k}={v}"   for k, v in run_cfg['model_parameters'].items()]
            model_param_set = " ".join(model_param_set)
            
            for label_text, var, original_type in self.model_param_vars:
                v = var.get()
                if original_type is list:
                    if not v:
                        final_value = []
                    else:
                        items_as_strings = [item.strip() for item in v.split(',')]
                        processed_items = []
                        for item in items_as_strings:
                            try:
                                processed_items.append(int(item))
                            except ValueError:
                                try:
                                    processed_items.append(float(item))
                                except ValueError:
                                    processed_items.append(item)
                        run_cfg['model_parameters'][label_text] = processed_items
                
                elif original_type is bool:
                    run_cfg['model_parameters'][label_text] = str(v).lower() == 'true'
                    
                else:
                    run_cfg['model_parameters'][label_text] = v
        
        
            cmd = 'nohup'
            log_route = GUI_DIR + '/GUI_log.txt'
            # log_route = 'GUI/GUI_log.txt'
            exe_file = "train.py"
        
        
            params_set = f" settings={run_cfg['model']}_settings" + \
            f" mode={run_cfg['mode']}" + \
            f" model={run_cfg['model']}" + \
            f" problem={run_cfg['problem']}" + f" cuda=[{self.device_var.get()[-1]}]" + \
            f" scale={run_cfg['scale']}" + \
            f" batch_size={run_cfg['batch_size']}" + \
            f" episodes={run_cfg['episodes']}" + \
            f" max_epochs={run_cfg['max_epochs']}" + \
            f" ckpt_path={run_cfg['ckpt_path']}" + \
            f" val_data_path={run_cfg['val_data_path']}"   
            #   f" test_data_path={test_cfg['test_data_path']}" + \
        
            exe_command = (
                f"cd {ROOT_DIR} && "
                f"({cmd} "
                f"{PYTHON_ROUTE} "
                f"{exe_file} "
                f" {params_set}"
                f" {model_param_set} "
                f"> {log_route} 2>&1 "
                "& echo $!)")
         


            self.log_path = log_route
            print('Running ')
            print(exe_command)
            command = exe_command
            stdin, stdout, stderr = self.client.exec_command(command)
            self.pid = stdout.readline().strip()  # 这里会直接得到PID
            print(f"PID: {self.pid}")
            stdout.channel.set_combine_stderr(True)
            self.is_running = True
            
        
            threading.Thread(
                target=self.monitor_log,
                args=(self.log_path,),
                daemon=True
            ).start() ##!!!
            
            
        else:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.update_log_display(f"[{current_time}]  Please select a valid mode (test/train).\n")
        
        
        
    
    def stop_click(self):
        
        exe_command = f"kill -9 {self.pid}"
        command = exe_command
        stdin, stdout, stderr = self.client.exec_command(command)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.update_log_display(f"[{current_time}] Stopping the process...\n")
        self.is_running = False
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.update_log_display(f"[{current_time}] The process has been stopped\n")
        self.sftp = self.client.open_sftp()
        self.stdout.channel.close()
    
    

        
    def on_select_mode_click(self, Combobox):

        self.select_mode = Combobox.get()
        
        if self.select_mode == 'train':
            
            self.test_param.config(text='Train Parameter')
            
            self.decoder_strategy_label.grid_remove()
            self.decoder_strategy_entry.grid_remove()
            
            self.epoch_label = ttk.Label(self.test_entry_frame, text = 'Epoch',style="White.TLabel")
            self.epoch_label.grid(row = 5, column = 0, sticky = 'e', padx =5, pady = 3)
            self.epoch_var = tk.IntVar(value = 5)
            self.epoch=ttk.Entry(self.test_entry_frame, textvariable = self.epoch_var, width=15)
            self.epoch.grid(row = 5, column = 1, sticky = 'ew', padx = 5, pady = 3)
            
            self.dataset_btn.grid(row= 6, column = 0, sticky = 'nesw', padx =5, pady = 3)
            self.dataset.grid(row= 6, column = 1, sticky = 'ew', padx =5, pady = 3)
            
            self.ckpt_btn.grid(row= 7, column = 0, sticky = 'nesw', padx =5, pady = 3)
            self.ckpt.grid(row= 7, column = 1, sticky = 'ew', padx =5, pady = 3)
            
            self.curve = tk.Button(self.test_entry_frame, text = 'Curve', command = self.plot_curve,foreground='#273c75',bg='white',)
            self.curve.grid(row = 8, column = 0, sticky = 'nesw',padx=5, pady=3)
            CURVE_LIST = ['loss', 'score','eval_score']
            self.curve_var = tk.StringVar(value = 'loss')
            self.curve_box = ttk.Combobox(self.test_entry_frame, textvariable = self.curve_var, values = CURVE_LIST, state = 'readonly',width = 5)
            self.curve_box.grid(row = 8, column = 1, sticky = 'ew', padx = 5, pady = 3)
            self.curve_box.bind('<<ComboboxSelected>>', lambda event: self.on_select_curve_click(self.curve_box))
            
            if self.connected:
                self.model_param_show()
            else:
                self.log_area.config(state = 'normal')
                self.log_area.delete('1.0', tk.END)
                self.update_log_display("[Warnings] in case that Model Parameter shows, please connect to the server first.\n")
            
        if self.select_mode == 'test':
            
            self.test_param.config(text='Test Parameter')
            
            """self.val_episode_label .grid_remove()
            self.val_episode.grid_remove()"""
            
            if hasattr(self, 'epoch_label'):
            
                self.epoch_label.grid_remove()
                self.epoch.grid_remove()
                
            if hasattr(self, "curve"):
            
                self.curve.grid_remove()
                self.curve_box.grid_remove()
            
            
            
            self.dataset_btn.grid(row= 6, column = 0, sticky = 'nesw', padx =5, pady = 3)
            self.dataset.grid(row= 6, column = 1, sticky = 'ew', padx =5, pady = 3)
            
            self.ckpt_btn.grid(row= 7, column = 0, sticky = 'nesw', padx =5, pady = 3)
            self.ckpt.grid(row= 7, column = 1, sticky = 'ew', padx =5, pady = 3)
            
            if self.connected:
                self.model_param_show()
            else:
                self.log_area.config(state = 'normal')
                self.log_area.delete('1.0', tk.END)
                self.update_log_display("[Warnings] in case that Model Parameter shows, please connect to the server first.\n")
            
            
        
        
    # MainApp.on_select_dataset_click
    def on_select_dataset_click(self):


        
        self.dataset_flag = True
        self.current_dir = ROOT_DIR + "/data/datasets/"
        self.open_folder()
        
        


    # MainApp.on_select_ckpt_click
    def on_select_ckpt_click(self):


        
        self.ckpt_flag = True
        # self.current_dir = ROOT_DIR + "EasyNCO/results/train/" + self.select_model + '/'
        self.current_dir = ROOT_DIR + "/pretrained"
        self.open_folder()
        
    def update_method_label(self,value):
        
        self.method_label.config(text= value)
        
        if self.connected:
            self.local_settings = 'local_settings.yaml'
            settings_file = ROOT_DIR +"/settings/" + self.select_model + '_settings.yaml'
            #/public/home/luoqing/code/EasyNCO_v0621/EasyNCO/settings/pomo_settings.yaml
            #/public/home/luoqing/code/EasyNCO/EasyNCO_v0621/EasyNCO/settings/pomo_settings.yaml
            
            print(settings_file)
            self.sftp.get(settings_file, self.local_settings)
            self.model_param_show()
            
        else:
            self.log_area.config(state = 'normal')
            self.log_area.delete('1.0', tk.END)
            self.update_log_display("[Warnings] Please connect to the server first.\n")

        print(f'method {self.select_model} has been selected')
        
        
    def update_task_label(self,value):
        
        self.task_label.config(text= value)

        
        print(f'task {self.select_task} has been selected')
        
        


    
    def monitor_log(self, log_path):
        
 
        
        try:
            file_stat = self.sftp.stat(log_path)
            self.last_position = file_stat.st_size
            
            self.sftp.close()

            command = f"tail -F -n 0 {log_path}"
            stdin, self.stdout, stderr = self.client.exec_command(command)
            
            
            # self.summary_line = []
            # pattern = r"Score:\s*(-?\d+\.\d+),\s*Loss:\s*(-?\d+\.\d+)"
            # pattern = r"Completed\s+\d+/\s*\d+\(100\.0%\)\s+Score:\s*(-?\d+\.\d+),\s*Loss:\s*(-?\d+\.\d+)"
            pattern = rf"Completed\s+{self.episodes}/{self.episodes}\(100\.0%\)\s+Score:\s*(-?\d+\.\d+),\s*Loss:\s*(-?\d+\.\d+)"   
            pattern_eval = r"Eval Done.*?Epoch\s+\d+/\s*\d+:.*?Value\[(\d+\.\d+)\]"
            
            self.result = {}
            result_match = r'\s*(\w+_score)\s+([\d\.Ee+-]+)'



            while self.is_running:
                """if stdout.channel.recv_ready():
                    line = stdout.readline()
                    self.update_log_display(line)""" 
                
                
                if self.stdout.channel.recv_ready():
                    

                    # print(len(self.stdout))
                    flag = 0
                    
                    for line in self.stdout:
                    # Read one line. This readline() will *not* block because recv_ready() was True.
                        
                        
                        # filter out unwanted lines
                        if "warning" in line.lower():
                            continue
                        if "testing" in line.lower():
                            continue
                        
                        if "utils.utils" in line.lower():
                            self.summary_line.append(line)
                            continue
                        
                        if "easynco.phases" in line.lower() and self.summary_flag:
                            
                            self.summary_transform()
                            self.summary_flag = False
                            
                            
                        #if re.search(r'^\s*[-_]+\s*$', line):
                        if re.match(r'^\s*[\|\-\_─]+\s*$', line):
                            line = '-' * 60+'\n'
                        
                        if re.search(result_match, line):
                            self.update_log_display(line,color='red')
                            continue 
                        
                        
                        self.update_log_display(line)
                        
                                
                        # match = re.search(pattern, line)
                        if re.search(pattern, line):
                            match = re.search(pattern, line)
                            score, loss = match.groups()
                            self.score.append(float(score))
                            self.loss.append(float(loss)) 
                            
                        # eval_match = re.search(pattern_eval, line)
                        if re.search(pattern_eval, line):
                            eval_match = re.search(pattern_eval, line)
                            val = float(eval_match.group(1))
                            self.eval_score.append(val)
                       
                            
                        if re.search(result_match, line):
                            self.result[match.group(1)] = float(match.group(2))
                        

                          

                else:
                    
                    time.sleep(0.1)
                    

                    
        except Exception as e:
            self.log_area.insert(tk.END, f"Monitoring Error: {str(e)}\n")
            # self.stop_monitoring()
            
            
    def update_log_display(self, text, color="black"):
 
        self.console_output_frame.after(0, self._safe_update_display, text, color)
        
        
    def update_summary_display(self, text):
        self.chart_output_frame.after(0, self._safe_update_summary_display, text)
    
            
    def _safe_update_display(self, text, color="black"):
        
        tag_name = f"log_color_{color}"
        if tag_name not in self._defined_tags:
            self.log_area.tag_configure(tag_name, foreground=color)
            self._defined_tags.add(tag_name)
    
        # 若该颜色 tag 未定义，动态定义
        try:
            self.log_area.tag_cget(tag_name, "foreground")
        except tk.TclError:
            self.log_area.tag_configure(tag_name, foreground=color)
        
        self.log_area.config(state = 'normal')
        # self.log_area.insert(tk.END, text)

        insert_index = self.log_area.index(tk.END)
        self.log_area.insert(tk.END, text)
        end_index = self.log_area.index(tk.END)
        self.log_area.tag_add(tag_name, insert_index, end_index)
        
        
        self.log_area.see(tk.END)
        self.log_area.configure(state = 'disabled')
        
    def _safe_update_summary_display(self, text):
        self.summary_area.config(state = 'normal')
        self.summary_area.insert(tk.END, text)
        self.summary_area.see(tk.END)
        self.summary_area.configure(state = 'disabled')
        
        
    def update_login_method(self):
        if self.login_method.get() == "key":
            # 显示密钥输入，隐藏密码输入
            self.key_label.grid()
            self.key_entry.grid()
            self.password_label.grid_remove()
            self.password_entry.grid_remove()
        else:
            # 显示密码输入，隐藏密钥输入
            self.password_label.grid(row=5, column=0, sticky='e', padx=5, pady=3)
            self.password_entry.grid(row=5, column=1, sticky='ew', padx=5, pady=3)
            self.key_label.grid_remove()
            self.key_entry.grid_remove()
            
            
            
    def on_enter(self,combobox):
        value = combobox.get()
        self.tooltip = tk.Toplevel(combobox)
        self.tooltip.wm_overrideredirect(True)
        x = combobox.winfo_rootx()
        y = combobox.winfo_rooty() + combobox.winfo_height()
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=value, background="#ffffe0", relief='solid', borderwidth=1,bg='white')
        label.pack()

    def on_leave(self,combobox):
        if hasattr(self, 'tooltip'):
            self.tooltip.destroy()
            
    def open_github(self):
        webbrowser.open_new("https://github.com/changliang5811/EasyNCO")
        
        
    def open_document(self):
        webbrowser.open_new("https://nco-demo.readthedocs.io/en/latest/")   
        
    
        
    def _list_remote_dir(self, directory):
        try:
            files = self.sftp.listdir(directory)
            return [
                (f, "dir" if self._is_remote_dir(posixpath.join(directory, f)) else "file")
                for f in files if not f.startswith('.')
            ]
        except Exception as e:
            print(f"Error listing directory: {e}")
            return []
        
    def _is_remote_dir(self, path):
        try:
            mode = self.sftp.stat(path).st_mode
            print(f"mode:{mode}")
            return stat.S_ISDIR(mode)
        except Exception as e:
            # 可以添加日志记录错误信息
            return False
        
    def open_folder(self):
        """创建GUI浏览远程目录"""
        popup = tk.Toplevel()
        popup.title(f"SFTP: {self.current_dir}")
        
        # 目录导航栏
        nav_frame = ttk.Frame(popup)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(nav_frame, text="←", command=lambda: self._navigate_up(popup)).pack(side=tk.LEFT)
        ttk.Label(nav_frame, text=self.current_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 文件列表
        tree = ttk.Treeview(popup, columns=("Type"), show="tree headings")
        tree.heading("#0", text="Name")
        tree.heading("Type", text="Type")
        tree.column("Type", width=80)
        
        # 填充初始目录内容
        for name, ftype in self._list_remote_dir(self.current_dir):
            icon = "📁" if ftype == "dir" else "📄"
            tree.insert("", tk.END, text=f"{icon} {name}", values=(ftype,))
        
        tree.bind("<Double-1>", lambda e: self._on_double_click(tree, popup))
        tree.pack(fill=tk.BOTH, expand=True)
        
        # 确认按钮
        btn_frame = ttk.Frame(popup)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(btn_frame, text="Confirm", command=lambda: self._on_confirm(tree, popup)).pack(side=tk.RIGHT)

    def _navigate_up(self, popup):
        """返回上一级目录"""
        new_dir = posixpath.dirname(self.current_dir)
        if new_dir != self.current_dir:
            self.current_dir = new_dir
            popup.destroy()
            self.open_folder()

    def _on_double_click(self, tree, popup):
        """双击进入子目录"""
        selected_item = tree.focus()
        if not selected_item:
            return
        
        item_text = tree.item(selected_item, "text")
        item_type = tree.item(selected_item, "values")[0]
        
        if not item_text or item_type != "dir":
            return
        
        # 提取目录名（去除图标）
        dir_name = item_text.split(" ", 1)[1]
        new_dir = posixpath.join(self.current_dir, dir_name)
        
        # 更新当前目录并刷新
        self.current_dir = new_dir
        popup.destroy()
        self.open_folder()

    def _on_confirm(self, tree, popup):
        """确认选择当前目录"""
        
        selected_item = tree.focus()
        if not selected_item:
            return
        
        item_text = tree.item(selected_item, "text")
        item_type = tree.item(selected_item, "values")[0]
        
        self.selected_path = self.current_dir
        
        if self.dataset_flag:
            self.selected_dataset_file = posixpath.join(self.current_dir, item_text.split(" ", 1)[1]) 
            #self.selected_dataset_file = posixpath.join(self.current_dir, item_text.split(" ", 1)[1]) if item_type == "file" else self.current_dir
            self.val_datasets_var.set(self.selected_dataset_file)
            self.dataset_flag = False
            self.insert_text_in_log(f"selected dataset file:  {self.selected_dataset_file}") 
            print(f"selected dataset path:{self.selected_dataset_file}")
            
        if self.ckpt_flag:
            self.selected_ckpt_file = posixpath.join(self.current_dir, item_text.split(" ", 1)[1]) if item_type == "file" else self.current_dir
            self.val_ckpt_var.set(self.selected_ckpt_file)
            self.ckpt_flag = False
            self.insert_text_in_log(f"selected ckpt file:   {self.selected_ckpt_file}") 
            print(f"selected ckpt path:{self.selected_ckpt_file}")
            
        popup.destroy()
        
    def insert_text_in_log(self, str):
        self.log_area.config(state = 'normal')
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_area.insert(tk.END, f"[{current_time}]  " +str+ "\n")
        self.log_area.config(state = 'disabled')
        
        
    # Canvas configure event handler
    def on_canvas_configure(self, event):
        # Update the scrollable frame's width to match the canvas's width
        # This prevents horizontal scrolling if the content is too wide for the canvas
        # self.param_canvas.itemconfig(self.param_canvas_frame_id, width=event.width)
        self.param_scrollable_frame.update_idletasks()
        self.param_canvas.itemconfig(self.param_canvas_frame_id, width=max(180, self.param_scrollable_frame.winfo_reqwidth()))


    # Frame configure event handler (when scrollable_frame changes size)
    def on_frame_configure(self, event):
        # Update the scrollregion of the canvas to match the scrollable_frame's bbox
        # This tells the canvas how large its scrollable area is
        self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all"))



       
        
    def draw_centered_image(self, event):
        self.summary_canvas.delete("image")  # 清除旧图像
        self.summary_canvas.create_image(
        event.width / 2,
        event.height / 2,
        image=self.photo,
        anchor=tk.CENTER,
        tags="image"
    )
        
    def model_param_show(self):
        
        self.param_canvas = tk.Canvas(self.model_frame,background="white",width=180) # Canvas can have its own background
        self.param_canvas.grid(row=0, column=0, sticky="nsew")
        # self.param_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.param_scrollbar = ttk.Scrollbar(self.model_frame, orient="vertical", command=self.param_canvas.yview)
        self.param_scrollbar.grid(row=0, column=1, sticky="nsew")
        self.param_canvas.configure(yscrollcommand=self.param_scrollbar.set)
        
        self.param_scrollbar_x = ttk.Scrollbar(self.model_frame, orient="horizontal", command=self.param_canvas.xview)
        self.param_scrollbar_x.grid(row=1, column=0, sticky="ew")
        self.param_canvas.configure(xscrollcommand=self.param_scrollbar_x.set)
        
        self.param_canvas.grid_rowconfigure(0,weight=1)
        self.param_canvas.grid_columnconfigure(0,weight=1)
        
        self.param_scrollable_frame = ttk.Frame(self.param_canvas, style="White.TLabelframe") # Apply style if desired
        self.param_scrollable_frame.grid(row=0,column=0, sticky="nsew")
        self.param_canvas_frame_id = self.param_canvas.create_window((0, 0), window=self.param_scrollable_frame, anchor="nw")
        
        self.param_scrollable_frame.grid_columnconfigure(0, weight=1)
        # self.param_scrollable_frame.grid_rowconfigure(0, weight=1)
        # self.param_scrollable_frame.grid_columnconfigure(1, weight=1)
        
        
        self.param_scrollable_frame.bind("<Configure>", self.on_frame_configure)
        self.param_canvas.bind("<Configure>", self.on_canvas_configure)
        
        

        ################################################
        # load yaml file and get additional parameters
        ###############################################
        
        
        with open(self.local_settings, 'r') as f:
            self.settings_cfg = yaml.safe_load(f)
            print(self.settings_cfg['model'])

            
        items = []
        
        items = flatten_dict(self.settings_cfg, exclude_keys=EXCLUDE_KEYS)
        
        
                    
        self.additional_vars = dict(items)

        self.additional_vars = self.match_gui_yaml(self.additional_vars, UTILS_YAML)
        
        
        var_num = len(self.additional_vars)
        keys_list = list(self.additional_vars.keys())
        
        var_num = len(self.additional_vars)
        keys_list = list(self.additional_vars.keys())
        self.model_param_vars = []
        self.additional_tkvars = {}
        
        for i in range(var_num): # Rows 5 to 10 (total 6 more rows)
            label_text = keys_list[i] # param_1, param_2, etc.
            
            adjust_frame = ttk.Frame(self.param_scrollable_frame, style="White.TFrame")
            adjust_frame.grid(row=i, column=0, sticky='nsew', padx=5, pady=3)
            
            adjust_frame.grid_columnconfigure(0, weight=1)
            adjust_frame.grid_rowconfigure(0, weight=1)
            adjust_frame.grid_rowconfigure(1, weight=1)
            
            
            ttk.Label(adjust_frame, text=label_text, style="White.TLabel").grid(row=0, column=0, sticky='w', padx=5, pady=3)
            
            value_frame = ttk.Frame(adjust_frame, style="White.TFrame")
            value_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=3)
            
            value_frame.grid_columnconfigure(0, weight=1)
            value_frame.grid_columnconfigure(1, weight=5)
            
            original_value = self.additional_vars[label_text]
            original_type = type(original_value)
    
            if isinstance(self.additional_vars[label_text], list):
                var_label = "list"
                list_as_string = ", ".join(map(str, self.additional_vars[label_text]))
                var = tk.StringVar(value=list_as_string)

            elif isinstance(self.additional_vars[label_text], bool):
                var_label = "bool"
                var = tk.StringVar(value="True" if self.additional_vars[label_text] else "False")
            
            elif isinstance(self.additional_vars[label_text], int):
                var_label = "int"
                var = tk.IntVar(value=self.additional_vars[label_text])
                print(self.additional_vars[label_text])
                
            elif isinstance(self.additional_vars[label_text], float):
                var_label = "float"
                var = tk.DoubleVar(value=self.additional_vars[label_text])
            
            else:
                var_label = "str"
                var = tk.StringVar(value=self.additional_vars[label_text])  # Default value for string variables
                

            ttk.Entry(value_frame, textvariable=var, state='normal',width = 18,style="White.TEntry").grid(row=0, column=1, sticky='nw', padx=5, pady=3)
                
            # self.additional_tkvars[label_text] = var
                
            # ttk.Entry(value_frame, textvariable=var, state='disable',width = 10).grid(row=0, column=1, sticky='nsew', padx=5, pady=3)
            ttk.Label(value_frame, text=var_label,style="White.TLabel",font=('Segoe UI', 10), foreground="#273c75",).grid(row=0, column=0, sticky='ne', padx=5, pady=3)  # Empty label for spacing
            # self.model_param_vars.append((label_text,var)) # Keep a reference if needed
            self.model_param_vars.append((label_text, var, original_type))
            
            
    def update_summary_test(self):
        self.summary_canvas.pack_forget()
        
        if hasattr(self, 'summary_area'):
            self.summary_area.pack_forget()
        
        self.summary_area = scrolledtext.ScrolledText(
            self.chart_output_frame,
            wrap = tk.WORD,
            state = 'disabled',
            bg = 'white',
            fg = 'black',
            insertbackground = 'black',
            font = ('Consolas', 10),
            borderwidth=1,
            relief="flat",
            width=1,
            height=1, 
        )
        
        self.summary_area.pack(padx = 5, pady = 5, fill = tk.BOTH, expand = True)
        
    
    def summary_transform(self):
        
        changable_list = ["${scale}","${problem}",
                          "${model}", "${batch_size}",
                          "${episodes}","${val_episode}",
                          "${decoder_strategy}",
                          "${save_top_k}","${every_n_epochs}",
                          "${max_epochs}","${log_every_n_steps}",
                          "${precision}","${enable_progress_bar}"]
        
        title_len = 109
        parameter_len = 35
        type_len =27
        value_len = 32
        
        self.update_summary_display("="*title_len+"\n")
        title = "  Summary of Parameters   "
        self.update_summary_display(f"{title}".center(title_len, " ")+"\n")
        self.update_summary_display("="*title_len+"\n")
        
        
        param_idx = " "
        param = "Parameter"
        param_type = "Source "
        value_str = "Value"
        
        self.update_summary_display(f" {param_idx:<3} | {param:<{parameter_len}} | {param_type:<{type_len}} | {value_str:<{value_len}} |\n")
        self.update_summary_display("-"*title_len+"\n")
        
        param_idx = 0
        format_lines = []
        
        for line in self.summary_line:
            
            if "- logger" in line:
                match = re.search(r"logger:\s*(\{.*\})", line)
                dict_str = match.group(1)
                logger_dict = ast.literal_eval(dict_str)
                for key, value in logger_dict.items():
                    param = f"logger.{key}"
                    param_type = "logger"
                    value_str = str(value)
                    
                    
                    # =======================================================
                    wrap_len = value_len
                    wrapped_lines = textwrap.wrap(value_str, width=wrap_len)
                    for i, wrapped in enumerate(wrapped_lines):
                        if i == 0:
                            format_line = f" {param_idx:<3} | {param:<{parameter_len}} | {param_type:<{type_len}} | {wrapped:<{wrap_len}} |"
                        else:
                            format_line = f"     | {'':<{parameter_len}} | {'':<{type_len}} | {wrapped:<{wrap_len}} |"         
                        # format_line = f" {param_idx:<3} | {param:<32} | {param_type:<11} | {value_str:<27} |"
                        format_lines.append(format_line)
                    # ===========================================================
                        
                    param_idx += 1
                    
            
            elif "- settings." in line:
                match = re.search(r"settings\.(\w+):\s*(\{.*\})", line)
                if match:
                    section, dict_str = match.groups()
                    config_dict = ast.literal_eval(dict_str)
                    for key, value in config_dict.items():
                        # 替换 changable 值
                        if isinstance(value, str) and value in changable_list:
                            for other_line in self.summary_line:
                                if f"- {value[2:-1]}:" in other_line:
                                    value = other_line.split(":")[-1].strip()
                        if key == "filename":
                            if isinstance(value, str):
                                value = value.replace("${model}", str(self.select_model))
                                value = value.replace("${problem}", str(self.select_task))
                                value = value.replace("${scale}", str(self.problem_size_var.get()))
                                config_dict["filename"] = value  
                        # param = f"{section}.{key}"
                        param = f"{key}"
                        param_type = f"settings.{section}"
                        value_str = str(value)
                        
                        # =======================================================
                        wrap_len = value_len
                        wrapped_lines = textwrap.wrap(value_str, width=wrap_len)
                        for i, wrapped in enumerate(wrapped_lines):
                            if i == 0:
                                format_line = f" {param_idx:<3} | {param:<{parameter_len}} | {param_type:<{type_len}} | {wrapped:<{wrap_len}} |"
                            else:
                                format_line = f"     | {'':<{parameter_len}} | {'':<{type_len}} | {wrapped:<{wrap_len}} |"         
                            # format_line = f" {param_idx:<3} | {param:<32} | {param_type:<11} | {value_str:<27} |"
                            format_lines.append(format_line)
                        # ===========================================================
                        
                        
                        # format_line = f" {param_idx:<3} | {param:<32} | {param_type:<11} | {value_str:<27} |"
                        # format_lines.append(format_line)
                        param_idx += 1
                    
                    
            else:
                match = re.search(r"- (\w+):\s*(.+)", line)
                if match:
                    key, value = match.groups()
                    param = key
                    param_type = "cfg"
                    value_str = value
                    
                    # =======================================================
                    wrap_len = value_len
                    wrapped_lines = textwrap.wrap(value_str, width=wrap_len)
                    for i, wrapped in enumerate(wrapped_lines):
                        if i == 0:
                            format_line = f" {param_idx:<3} | {param:<35} | {param_type:<{type_len}} | {wrapped:<{wrap_len}} |"
                        else:
                            format_line = f"     | {'':<35} | {'':<{type_len}} | {wrapped:<{wrap_len}} |"         
                        # format_line = f" {param_idx:<3} | {param:<32} | {param_type:<11} | {value_str:<27} |"
                        format_lines.append(format_line)
                     # ===========================================================
                     
                    # format_line = f" {param_idx:<3} | {param:<32} | {param_type:<11} | {value_str:<27} |"
                    # format_lines.append(format_line)
                    param_idx += 1
                    
                    
        for line in format_lines:
            self.update_summary_display(f"{line}\n")
            # print(f'self.update_summary_display("{line}")')     
                
        
        
        self.update_summary_display("-"*title_len+"\n")
        
        
    def on_select_curve_click(self, Combobox):
        self.select_curve = Combobox.get()
        print(f"Curve {self.select_curve} has been selected")
        
    
    def plot_curve(self):
        
        if not self.score or not self.loss or not self.eval_score:
            return
        
        self.select_curve = self.curve_box.get()
        # self.steps_per_epoch = math.ceil(self.episodes / self.batch_size)
        self.steps_per_epoch = 1

        if self.select_curve == 'loss':
    
            # 弹出新窗口
            plot_win = tk.Toplevel(self)
            plot_win.title(f"Training Loss Curve")

            fig, ax = plt.subplots(figsize=(8, 4))

            # 计算 epoch 坐标
            epochs = [i / self.steps_per_epoch + 1 for i in range(len(self.loss))]
            
            
            ax.plot(epochs, self.loss, label="Loss", color="red")

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f" {self.select_model}-{self.select_task}-{self.scale} Training Loss Curve")
            ax.grid(True)
            ax.legend()

            #设置x轴刻度为整数（每个epoch）
            # ax.set_xticks(range(int(max(epochs)) + 1))
            # 设置自适应x轴刻度
            max_epoch = int(max(epochs))
            if max_epoch <= 10:
                tick_step = 1
            elif max_epoch <= 100:
                tick_step = 10
            else:
                tick_step = 100
                
            
            ax.set_xticks(range(0, max_epoch + 1, tick_step))


            plt_canvas = FigureCanvasTkAgg(fig, master=plot_win)
            plt_canvas.draw()
            plt_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        elif self.select_curve == 'eval_score':
            # 弹出新窗口
            plot_win = tk.Toplevel(self)
            plot_win.title(f"Training Eval_Score Curve")

            fig, ax = plt.subplots(figsize=(8, 4))

            # 计算 epoch 坐标
            epochs = [i / self.steps_per_epoch for i in range(len(self.eval_score))]
            
            ax.plot(epochs, self.eval_score, label="Eval_Score", color="green")

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Eval_Score")
            ax.set_title(f" {self.select_model}-{self.select_task}-{self.scale} Training Eval_Score Curve")
            ax.grid(True)
            ax.legend()

            #设置x轴刻度为整数（每个epoch）
            # ax.set_xticks(range(int(max(epochs)) + 1))
            max_epoch = int(max(epochs))
            if max_epoch <= 10:
                tick_step = 1
            elif max_epoch <= 100:
                tick_step = 10
            else:
                tick_step = 100

                
            ax.set_xticks(range(0, max_epoch + 1, tick_step))

            plt_canvas = FigureCanvasTkAgg(fig, master=plot_win)
            plt_canvas.draw()
            plt_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
            
        else:
    
            # 弹出新窗口
            plot_win = tk.Toplevel(self)
            plot_win.title(f"Training Score Curve")

            fig, ax = plt.subplots(figsize=(8, 4))

            # 计算 epoch 坐标
            epochs = [i / self.steps_per_epoch + 1 for i in range(len(self.score))]
            
            ax.plot(epochs, self.score, label="Score", color="blue")

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.set_title(f" {self.select_model}-{self.select_task}-{self.scale} Training Score Curve")
            ax.grid(True)
            ax.legend()

            #设置x轴刻度为整数（每个epoch）
            # ax.set_xticks(range(int(max(epochs)) + 1))
            max_epoch = int(max(epochs))
            if max_epoch <= 10:
                tick_step = 1
            elif max_epoch <= 100:
                tick_step = 10
            else:
                tick_step = 100
            
            ax.set_xticks(range(0, max_epoch + 1, tick_step))

            plt_canvas = FigureCanvasTkAgg(fig, master=plot_win)
            plt_canvas.draw()
            plt_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def print_watermark(self):
        
        self.log_area.config(state = 'normal')
        self.log_area.delete('1.0', tk.END)
        
        
        for line in WATERMARK:
            self.log_area.insert(tk.END, line)
        
        # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # self.log_area.insert(tk.END, f"[{current_time}]  Starting the porcess...\n")

        
        self.log_area.config(state = 'disabled')
        
        
    def is_pid_running(self):

        if not self.pid:
            return False # No PID to check

        check_command = f"ps -p {self.pid} -o pid= | grep -w {self.pid} | wc -l"

        try:
            pid_stdin, pid_stdout, pid_stderr = self.client.exec_command(check_command)
            output = pid_stdout.read().decode().strip()
            error_output = pid_stderr.read().decode().strip()

            if error_output:
                # Handle cases where ps might not work or permission issues
                print(f"Error checking PID {self.pid}: {error_output}")
                return False

            if output == '1': # For the `ps -p ... | grep ... | wc -l` command
                print("is running")
                return True
            elif output == 'running': # For the `kill -0` command
                print("is running")
                return True
            else:
                return False # PID not found

        except paramiko.SSHException as e:
            print(f"SSH check PID command failed: {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during PID check: {e}")
            return False
        

    """def monitor_pid(self):

        if not self.pid:
            print("No PID set to monitor.")
            return

        if not self.is_pid_running():
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.update_log_display(f"[{current_time}]  Process with PID {self.pid} has finished.\n")
            
            self.is_running  = False
            
            # 记得关闭上一个管道
            self.sftp = self.client.open_sftp()
            self.stdout.channel.close()"""
            
        
    def monitor_pid(self):
        

 
        # 方法2：远程进程检查（通过SSH）
        if hasattr(self, "client") and hasattr(self, "pid"):
            stdin, pid_stdout, stderr = self.client.exec_command(f"ps -p {self.pid}")
            print("Runing False 111")
            
            # if not pid_stdout.read().decode().strip():
            if not self.is_pid_running():
                    
                print("Runing False")
                self.is_running = False
                    
                # 记得关闭上一个管道
                self.sftp = self.client.open_sftp()
                self.stdout.channel.close()
                    
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.update_log_display(f"[{current_time}]  Process with PID {self.pid} has finished.\n")


    def update_method_listbox(self, method_list: list):
        self.methodContent.delete(0, 'end')
        for method in method_list:
            self.methodContent.insert('end', method)
            
            
    def match_gui_yaml(self, param_dict, utils_yaml):
        """param_dict: from settings.yaml, pull from severs  """
        """utils_yaml: for GUI, from ../utils/utils_gui_params.yaml"""
        task_param = return_task_method_dict(self.select_task, self.select_model, file=utils_yaml)
        print("task_param:", task_param)
        if task_param == "None":
            return param_dict
        
        if "train" in task_param or "test" in task_param:
            print(task_param.keys())
            task_param = task_param[self.select_mode]
            

        for key, value in task_param.items():
            if key in param_dict:
                param_dict[key] = value

                
        return param_dict
    
    
    def save_click(self):
        
        if not self.connected:
            self.update_log_display("[Warnings]  Please connect to server first.\n")
            return
        


        self._search_train_logs_gui(mode = self.select_mode)
        
    
    
    def _search_train_logs_gui(self, mode):
        """
        启动专门搜索 log/train 目录下 train.log 文件的 GUI。
        """
        search_popup = tk.Toplevel(self, bg='white')
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.update_log_display(f"[{current_time}]  Searching '{mode}.log' files... in remote server\n")
        
        search_popup.title(f"Search {mode} log files")
        search_popup.grab_set() # Make this window modal

        status_label = ttk.Label(search_popup, text=f"Searching for '{mode}.log' files in 'results/'...", font=('Segoe UI', 10))
        status_label.pack(pady=20, padx=20)

        log_files_found = []

        # 在单独的线程中执行搜索，防止GUI卡死
        search_thread = threading.Thread(target=lambda: self._execute_train_log_search_and_display(search_popup, status_label, log_files_found))
        search_thread.start()
    
    
    def _execute_train_log_search_and_display(self, search_popup, status_label, log_files_found):
        """
        执行特定路径的 log 文件搜索并更新 GUI。
        """
        # /public/home/luoqing/code/EasyNCO_v0708/EasyNCO/results/train/matpoenet_atsp/2025-07-22-19-17-28_matpoenet_atsp_20_train/eval.log
        # EasyNCO/results/train/matpoenet_atsp/2025-07-22-19-17-28_matpoenet_atsp_20_train/eval.log
        target_base_dir = ROOT_DIR + "/results"

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.update_log_display(f"[{current_time}]  Searching for '{self.select_mode}.log' files in  'results/'...\n")

        # 确保 SFTP 连接存在且 target_base_dir 存在
        try:
            if not self.sftp:
                raise Exception("SFTP client not connected.")
            # 尝试访问目标路径以确保其存在
            self.sftp.stat(target_base_dir)
            if self.select_mode == "test":
                self._search_log_files(target_base_dir, log_files_found, target_filename="eval.log")
            else:
                self._search_log_files(target_base_dir, log_files_found, target_filename="train.log")
                
        except FileNotFoundError:
            self.insert_text_in_log(f"Remote path '{target_base_dir}' not found.\n")
            log_files_found.clear() # Clear results if base path is not found
        except Exception as e:
            self.insert_text_in_log(f"Error accessing remote path '{target_base_dir}': {e}")
            log_files_found.clear()


        # 搜索完成后，调度到主线程更新GUI
        search_popup.after(0, lambda: self._display_selected_log_files(search_popup, status_label, log_files_found))
    
    def _search_log_files(self, remote_path, log_files_found, log_file_extensions=None, target_filename=None):
        """
        递归搜索远程目录中的文件。
        :param remote_path: 当前要搜索的远程路径。
        :param log_files_found: 存储找到的文件路径的列表。
        :param log_file_extensions: 要搜索的文件的扩展名元组 (例如 ('.log', '.txt'))。
        :param target_filename: 特定要搜索的文件名 (例如 'train.log')。如果设置，则忽略 log_file_extensions。
        """

        try:
            if not self.sftp:
                return

            for entry in self.sftp.listdir_attr(remote_path):
                full_remote_path = posixpath.join(remote_path, entry.filename)
                if entry.st_mode is not None:
                    if stat.S_ISDIR(entry.st_mode):
                        self._search_log_files(full_remote_path, log_files_found, log_file_extensions, target_filename)
                    elif stat.S_ISREG(entry.st_mode):
                        # 检查文件名是否符合要求
                        is_target_file = False
                        if target_filename and entry.filename == target_filename:
                            # log_files_found.append(full_remote_path)
                            is_target_file = True
                        elif log_file_extensions and any(entry.filename.lower().endswith(ext) for ext in log_file_extensions):
                            # log_files_found.append(full_remote_path)
                            is_target_file = True
                            
                        if is_target_file:
                            found_eval_done = False
                            try:
                                self.insert_text_in_log(f"Checking content of remote file: {full_remote_path}")
                                # 打开远程文件
                                with self.sftp.open(full_remote_path, 'r', bufsize=4096) as f:
                                    for line in f:
                                        if "Eval Done" in line:
                                            found_eval_done = True
                                            break
                                    if found_eval_done:
                                        log_files_found.append(full_remote_path)
                                    # self.insert_text_in_log(f"Skipping '{full_remote_path}' (does not contain 'Eval Done').")
                            except Exception as file_read_e:
                                self.insert_text_in_log(f"Error reading remote file {full_remote_path}: {file_read_e}")
                            
                            
        except Exception as e:
            self.insert_text_in_log(f"Error searching remote path {remote_path}: {e}")
    
    def _show_file_preview(self, parent_window, file_path):
        """在一个新的弹窗中显示指定文件的内容。"""
        preview_popup = tk.Toplevel(parent_window)
        preview_popup.title(f"Preview: {os.path.basename(file_path)}")
        preview_popup.geometry("800x600")
        preview_popup.grab_set()

        st = scrolledtext.ScrolledText(preview_popup, wrap=tk.WORD, font=("Consolas", 10))
        st.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        st.insert(tk.INSERT, f"Reading remote file content...\n{file_path}\n\n")
        preview_popup.update_idletasks()
        
        try:
            content = self._read_remote_file_content(file_path)
            st.delete("1.0", tk.END)
            st.insert(tk.INSERT, content)
        except Exception as e:
            st.delete("1.0", tk.END)
            st.insert(tk.INSERT, f"Error reading remote file:\n\n{e}")

        st.config(state=tk.DISABLED)
       
        
    def _read_remote_file_content(self, remote_path):
        if not hasattr(self, 'sftp') or not self.sftp:
            error_message = "Error: SFTP connection is not active or available."
            print(error_message)
            return error_message

        print(f"--- Attempting to read remote file: {remote_path} ---")
    
        try:
            with self.sftp.open(remote_path, 'r', bufsize=4096) as f:
                content = f.read()
            return content

        except Exception as e:
            error_message = (
                f"Failed to read remote file: {os.path.basename(remote_path)}\n\n"
                f"Path: {remote_path}\n\n"
                f"Error details: {e}"
            )
            print(error_message)
            return error_message
 
    
    
    def _display_selected_log_files(self, search_popup, status_label, log_files_found):
        """
        在新的窗口中显示找到的log文件列表，每个文件带有勾选框。
        """
        status_label.destroy() # 销毁搜索状态标签
        search_popup.title(f"Select '{self.select_mode}.log' files to download")

        if not log_files_found:
            ttk.Label(search_popup, text=f"No '{self.select_mode}.log' files found under 'EasyNCO/result'.").pack(pady=20)
            ttk.Button(search_popup, text="Close", command=search_popup.destroy).pack(pady=10)
            search_popup.grab_release()
            return

        list_frame = ttk.Frame(search_popup, style = "White.TFrame")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(list_frame)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style = "White.TFrame")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.configure(bg='white')

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.selected_log_files = {} # 重置选中的文件字典

        for log_file_path in sorted(log_files_found):
            
            path_parts = log_file_path.split('/')
            if len(path_parts) >= 2:
                display_text = f"{path_parts[-2]}/{path_parts[-1]}"
            else:
                display_text = log_file_path
                
            item_frame = ttk.Frame(scrollable_frame, style="White.TFrame")
            item_frame.pack(anchor=tk.W, fill=tk.X, padx=5, pady=2)
            
            var = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(item_frame, text=display_text, variable=var, style="White.TCheckbutton")
            chk.pack(side=tk.LEFT)
            

            preview_button = tk.Button(
                item_frame, 
                text="🔍",  
                width=3,     
                bg='white',
                foreground='#273c75',
                relief="flat",
                command=lambda path=log_file_path: self._show_file_preview(search_popup, path)
            )
            preview_button.pack(side=tk.RIGHT, padx=(0, 5))
            
            
            self.selected_log_files[log_file_path] = var

        confirm_btn_frame = ttk.Frame(search_popup, style = "White.TFrame")
        confirm_btn_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(confirm_btn_frame, 
                   text="Confirm Download",
                   font=('Segoe UI', 10, 'bold'), 
                   bg='#273c75',
                   foreground='white', 
                   command=lambda: self._confirm_download_log_files(search_popup)).pack(side=tk.RIGHT)
        search_popup.grab_release()
        
        
    def _confirm_download_log_files(self, search_popup):
        files_to_download = [
            file_path for file_path, var in self.selected_log_files.items() if var.get()
        ]

        if not files_to_download:
            messagebox.showinfo("No Files Selected", "Please select at least one log file to plot.", parent=search_popup)
            return
        
        problem_types = set() # 使用集合来存储所有唯一的 "problem" 类型

        for file_path in files_to_download:
            try:
                # 1. 从路径中获取文件名
                path_obj = Path(file_path)
                problem_part = path_obj.parent.parent.name
                parts = problem_part.split('_')[-1]
                problem_types.add(parts)
                

            except IndexError:
                print(f"Warning: Could not parse problem type from filename: {file_path}")
                continue

        # 4. 检查集合中的 "problem" 类型数量
        if len(problem_types) > 1:
            message = (
                f"You have selected files from multiple problem types:\n\n"
                f"{sorted(list(problem_types))}\n\n"
                f"Plotting data from different problems together may lead to incorrect results.\n\n"
                f"Do you want to continue anyway?"
            )
            
            # 5. 弹出一个 "确定/取消" 对话框
            proceed = messagebox.showwarning(
                "Different Problem Types Detected",
                message,
                parent=search_popup
            )
            
            return

         # 获取当前 Python 文件的目录
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建 log_temp 文件夹的完整路径
        local_dir = os.path.join(current_script_dir, "log_temp")

        # 检查 log_temp 文件夹是否存在，如果不存在则创建它
        try:
            if os.path.exists(local_dir):
                self.insert_text_in_log(f"Clearing existing files in: {local_dir}")
                # 遍历并删除目录内的所有文件和子目录
                for item in os.listdir(local_dir):
                    item_path = os.path.join(local_dir, item)
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path) # 删除文件或软链接
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path) # 删除子目录及其内容

            os.makedirs(local_dir, exist_ok=True)
            self.insert_text_in_log(f"Saving log files to local directory: {local_dir}")
        except OSError as e:
            messagebox.showerror("Directory Creation Error", f"Failed to create directory '{local_dir}': {e}", parent=search_popup)
            self.insert_text_in_log(f"Error creating local directory: {e}")
            return



        download_success = True
        for remote_file_path in files_to_download:
            
            original_file_name = posixpath.basename(remote_file_path) 
            # "eval.log"
            parent_dir_path = posixpath.dirname(remote_file_path)
            parent_folder_name = posixpath.basename(parent_dir_path)
            local_file_name = f"{parent_folder_name}_{original_file_name}" 
            # "timestamp_model_eval.log"
            
            # local_file_name = posixpath.basename(remote_file_path)
            local_full_path = posixpath.join(local_dir, local_file_name)
            if not self._download_file(remote_file_path, local_full_path):
                download_success = False

        if download_success:
            messagebox.showinfo("Download Complete", "All selected log files downloaded successfully!", parent=search_popup)
        else:
            messagebox.showerror("Download Failed", "Some log files could not be downloaded. Check logs for details.", parent=search_popup)

        search_popup.destroy()

    def _download_file(self, remote_path, local_path):
        """
        通过SFTP下载单个文件。
        """
        try:
            if not self.sftp:
                self.insert_text_in_log("SFTP client not connected. Cannot download file.")
                # self.update_log_display("SFTP client not connected. Cannot download file.")
                return False

            self.sftp.get(remote_path, local_path)
            self.insert_text_in_log(f"Successfully downloaded: {remote_path} to {local_path}")
            # self.update_log_display(f"Successfully downloaded: {remote_path} to {local_path}")
            return True
        except Exception as e:
            self.insert_text_in_log(f"Error downloading {remote_path}: {e}")
            # self.update_log_display(f"Error downloading {remote_path}: {e}")
            return False
    
    def plot_click(self):
        
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        log_temp_dir = os.path.join(current_script_dir, "log_temp")
        
        if not os.path.exists(log_temp_dir) or not os.listdir(log_temp_dir):
            messagebox.showinfo("No Log Files", f"The '{log_temp_dir}' folder is empty or does not exist. Please download log files first.", parent=self.master)
            self.update_log_display(f"[Warnings]  The '{log_temp_dir}' folder is empty or does not exist. Please click \"Select\" first.\n")
            return
        
        
        log_file_names = []
        scores = []
        augmented_scores = []
        gaps = []
        augmented_gaps = []
        
        eval_done_pattern = re.compile(r"Eval Done\s*Score Summary: Avg: \[([\d\.-]+)\]*Avg Augmented: \[([\d\.-]+)\]")
        #Eval Done ==> Problem[tsp], Score Summary: Avg: 36321.0344, Avg Augmented: 36321.0344
        gap_pattern = re.compile(r"Eval Done.*Gap Summary: Avg: \[([\d\.-]+)\].*Avg Augmented: \[([\d\.-]+)\]")
        score_avg_pattern = r"Avg:\s*([\d.]+)"
        score_avg_aug_pattern = r"Avg Augmented:\s*([\d.]+)"
        gap_avg_pattern = r"Avg:\s*([\d.]+)%"
        gap_avg_aug_pattern = r"Avg Augmented:\s*([\d.]+)%"
        
        for filename in os.listdir(log_temp_dir):
            if filename.endswith(".log"):
                file_path = os.path.join(log_temp_dir, filename)
                self.update_log_display(f"Processing log file: {file_path}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            
                            if "Eval Done" in line:
                                if "Score Summary" in line:
                                    
                                    # 横坐标提取放在这里，因为Eval done 可能是多行
                                    base_name = filename.replace(".log", "")
                                    parts = base_name.split('_')
                                    if len(parts) >= 4:
                                        display_name = '_'.join(parts[:4])
                                    else:
                                        display_name = base_name
                                    log_file_names.append(display_name)
                                        
                                    
                                    
                                    # 提取 Avg Score
                                    match_avg = re.search(score_avg_pattern, line)
                                    if match_avg:
                                        scores.append(float(match_avg.group(1)))
                
                                    # 提取 Avg Augmented Score
                                    match_avg_aug = re.search(score_avg_aug_pattern, line)
                                    if match_avg_aug:
                                        augmented_scores.append(float(match_avg_aug.group(1)))

                                elif "Gap Summary" in line:
                                # 提取 Avg Gap
                                    match_avg = re.search(gap_avg_pattern, line)
                                    if match_avg:
                                        gaps.append(float(match_avg.group(1)))
                                    else:
                                        gaps.append(None)
                
                                # 提取 Avg Augmented Gap
                                    match_avg_aug = re.search(gap_avg_aug_pattern, line)
                                    if match_avg_aug:
                                        augmented_gaps.append(float(match_avg_aug.group(1)))
                                    else:
                                        augmented_gaps.append(None)
                                        
                                
                            """match = eval_done_pattern.search(line)
                            if match:
                                
                                base_name = filename.replace(".log", "")
                                parts = base_name.split('_')
                                if len(parts) >= 4:
                                    display_name = '_'.join(parts[:4])
                                else:
                                    display_name = base_name
                                log_file_names.append(display_name)

                                scores.append(float(match.group(1))) # Score 是第一个捕获组
                                augmented_scores.append(float(match.group(2))) # Augmented Score 是第二个捕获组
                                
                                gap_match = gap_pattern.search(line)
                                if gap_match:
                                    gaps.append(float(gap_match.group(1)))
                                    augmented_gaps.append(float(gap_match.group(2)))
                                else:
                                    gaps.append(None)
                                    augmented_gaps.append(None)
                                self.update_log_display(f"Extracted from {filename}: Score={match.group(1)}, Augmented Score={match.group(2)}")
                                break # 假设每份log文件只取第一条符合的Eval Done行，如果需要多条请移除此行"""
                except Exception as e:
                    self.update_log_display(f"Error reading or processing {filename}: {e}")

        if not log_file_names:
            messagebox.showinfo("No Data Found", "No 'Eval Done' lines with Score data found in any log files.", parent=self.master)
            self.update_log_display("Plotting failed: No relevant data extracted.")
            return
        
        self._create_plot_popup(log_file_names, scores, augmented_scores, gaps, augmented_gaps)
        self.update_log_display("Plotting complete. Graph displayed in new window.")
        
        
    def _create_plot_popup(self, log_file_names, scores, augmented_scores, gaps, augmented_gaps):
        """
        在新的 Tkinter Toplevel 窗口中创建 Matplotlib 柱状图并嵌入。
        窗口左侧有控制参数的 Entry，右侧是图表。
        """
        plot_popup = tk.Toplevel(self, bg='white')
        plot_popup.title("Metric Plot with Controls")
        plot_popup.geometry("1200x800") # 给予更宽敞的初始大小

        # --- 主框架，用于容纳侧边栏和绘图区 ---
        main_frame = ttk.Frame(plot_popup,style='White.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- 1. 左侧控制侧边栏 ---
        sidebar_frame = ttk.Frame(main_frame, width=250,style='White.TFrame')
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar_frame.pack_propagate(False) # 防止侧边栏缩放

        # --- 2. 右侧绘图区域 ---
        plot_area_frame = ttk.Frame(main_frame,style='White.TFrame')
        plot_area_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- 在侧边栏中创建 Entry 控件 ---
        # 使用 StringVar 来管理 Entry 的内容
        title_var = tk.StringVar()
        xlabel_var = tk.StringVar()
        ylabel_var = tk.StringVar()
        ylim_var = tk.StringVar() # 用于 y 轴范围, e.g., "0, 100"
        xtick_labels_var = tk.StringVar()

        # 使用 grid 布局来对齐标签和输入框
        controls_container = ttk.LabelFrame(sidebar_frame, text="Plot Parameters",style='White.TLabelframe')
        controls_container.pack(fill=tk.X, pady=5)

        ttk.Label(controls_container, text="Title:",style='White.TLabel').grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(controls_container, textvariable=title_var).grid(row=0, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(controls_container, text="X Label:",style='White.TLabel').grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(controls_container, textvariable=xlabel_var).grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(controls_container, text="Y Label:",style='White.TLabel').grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(controls_container, textvariable=ylabel_var).grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        
        ttk.Label(controls_container, text="Y-Lim (min,max):",style='White.TLabel').grid(row=3, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(controls_container, textvariable=ylim_var).grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        ttk.Label(controls_container, text="format: 1,10", foreground="gray", style='White.TLabel').grid(row=4, column=1, sticky="w", padx=5, pady=3)

        ttk.Label(controls_container, text="X-Tick Labels:",style='White.TLabel').grid(row=5, column=0, sticky="w", padx=5, pady=3)
        ttk.Entry(controls_container, textvariable=xtick_labels_var).grid(row=5, column=1, sticky="ew", padx=5, pady=3)
        ttk.Label(controls_container, text="format: ['a', 'b', ...]", foreground="gray", style='White.TLabel').grid(row=6, column=1, sticky="w", padx=5, pady=3)


        # 设置grid列的权重，使第二列（输入框）可以伸展
        controls_container.columnconfigure(1, weight=1)

        # --- 更新按钮 ---
        update_button = tk.Button(sidebar_frame, text="Update Plot", command=lambda: update_plot(), bg="#273c75",foreground='white',font=('Segoe UI', 10, 'bold'))
        update_button.pack(fill=tk.X, pady=10)

        # --- 在绘图区域顶部添加指标选择器 ---
        control_frame = ttk.Frame(plot_area_frame,style='White.TFrame')
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        ttk.Label(control_frame, text="Select Metric:",style='White.TLabel').pack(side=tk.LEFT, padx=5)
        metric_var = tk.StringVar(value="Score")
        metric_chooser = ttk.Combobox(control_frame, textvariable=metric_var,
                                      values=["Score", "Augmented Score","Gap","Augmented Gap"], state="readonly")
        metric_chooser.pack(side=tk.LEFT, padx=5)

        # --- Matplotlib 画布和工具栏 ---
        fig, ax = plt.subplots(figsize=(10, 6))
        canvas = FigureCanvasTkAgg(fig, master=plot_area_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, plot_area_frame)
        toolbar.update()
        canvas_widget.focus_set()

        # --- 核心更新函数 ---
        def update_plot(event=None):
            
            
            ax.clear()
              
            plot_xticklabels = log_file_names # 默认使用原始文件名
            user_labels_str = xtick_labels_var.get()
            
            if user_labels_str: 
                try:
                    parsed_labels = ast.literal_eval(user_labels_str)
                    
                    if isinstance(parsed_labels, list) and len(parsed_labels) == len(log_file_names):
                        plot_xticklabels = parsed_labels 
                    else:

                        
                        message = (
                        f"Input error!\n\n"
                        f"The number of x-tick is {len(log_file_names)}, \n\n"
                        f"Your input is {len(parsed_labels)} .\n\n"
                        f"reset to default value."
                        )
                        # 2. 显示警告弹窗
                        # 注意：请确保 plot_popup 是您的Toplevel窗口变量名
                        messagebox.showwarning("Input Format Error", message, parent=plot_popup)
                        
                        # 3. 将输入框内容重置为有效的默认值
                        xtick_labels_var.set(str(log_file_names))
                        # 使用默认标签继续绘图
                        plot_xticklabels = log_file_names
                        
                except (ValueError, SyntaxError) as e:
                    print(f"X-Tick Labels 语法无效：{e}")
                    xtick_labels_var.set(str(log_file_names)) # 重置为有效值
            
            selected_metric = metric_var.get()
            x = range(len(log_file_names))
            width = 0.5
            
            
            metrics_config = {
                "Score": {
                    "data": scores,
                    "label": "Score",
                    "color": "#c65574",
                    "default_title": "Score",
                    "default_ylabel": "Score Value"
                },
                "Augmented Score": {
                    "data": augmented_scores,
                    "label": "Augmented Score",
                    "color": "#3a94ae",
                    "default_title": "Augmented Score",
                    "default_ylabel": "Augmented Score Value"
                },
                "Gap": {
                    "data": gaps,
                    "label": "Gap",
                    "color": "#858FAC",
                    "default_title": "Gap",
                    "default_ylabel": "Gap Value (%)"
                },
                "Augmented Gap": {
                    "data": augmented_gaps,
                    "label": "Augmented Gap",
                    "color": "#2e8a79",
                    "default_title": "Augmented Gap",
                    "default_ylabel": "Augmented Gap Value (%)"
                }
            }


            all_default_titles = {v['default_title'] for v in metrics_config.values()}
            all_default_ylabels = {v['default_ylabel'] for v in metrics_config.values()}


            selected_config = metrics_config[selected_metric]

            current_title = title_var.get().strip()
            current_ylabel = ylabel_var.get().strip()

            if not current_title or current_title in all_default_titles:
                title_var.set(selected_config['default_title'])

            if not current_ylabel or current_ylabel in all_default_ylabels:
                ylabel_var.set(selected_config['default_ylabel'])

            if not xlabel_var.get().strip():
                xlabel_var.set('Method')

            bars = ax.bar(x, 
                        selected_config['data'], 
                        width, 
                        label=selected_config['label'], 
                        color=selected_config['color'])

            ax.bar_label(bars, fmt='%.4f', padding=3)

            """if selected_metric == "Score":
                bars = ax.bar(x, scores, width, label='Score', color = "#c65574")
                # 设置默认值
                if not title_var.get(): title_var.set('Score')
                if not ylabel_var.get(): ylabel_var.set('Score Value')
                # title_var.set('Score')
                # ylabel_var.set('Score Value')
            elif selected_metric == "Augmented Score": # "Augmented Score"
                bars = ax.bar(x, augmented_scores, width, label='Augmented Score', color='#3a94ae')
                # 设置默认值
                if not title_var.get(): title_var.set('Augmented Score')
                if not ylabel_var.get(): ylabel_var.set('Augmented Score Value')
                # title_var.set('Augmented Score')
                # ylabel_var.set('Augmented Score Value')
            elif selected_metric == "Gap":
                bars = ax.bar(x, gaps, width, label='Gap', color='#858FAC')
                # 设置默认值
                if not title_var.get(): title_var.set('Gap')
                if not ylabel_var.get(): ylabel_var.set('Gap Value')
                # title_var.set('Gap')
                # ylabel_var.set('Gap Value')
            else:
                bars = ax.bar(x, augmented_gaps, width, label='Augmented Gap', color='#2e8a79')
                # 设置默认值
                if not title_var.get(): title_var.set('Augmented Gap')
                if not ylabel_var.get(): ylabel_var.set('Augmented Gap Value')
                # title_var.set('Augmented Gap')
                # ylabel_var.set('Augmented Gap Value')
            
            if not xlabel_var.get(): xlabel_var.set('Method')

            ax.bar_label(bars, fmt='%.4f', padding=3)"""

            # --- 从 Entry (StringVar) 获取值并应用到图表 ---
            ax.set_title(title_var.get())
            ax.set_xlabel(xlabel_var.get())
            ax.set_ylabel(ylabel_var.get())

            self.temp_title_var = title_var.get()
            self.temp_xlabel_var = xlabel_var.get()
            self.temp_ylabel_var = ylabel_var.get()

            
            # 解析并设置 Y 轴范围，带错误处理
            try:
                ylim_text = ylim_var.get()
                if ylim_text:
                    min_val, max_val = map(float, ylim_text.split(','))
                    ax.set_ylim([min_val, max_val])
            except (ValueError, IndexError):
                print(f"Invalid Y-Lim format: '{ylim_var.get()}'. Please use 'min,max'.")
                # 如果格式错误，可以清除 ylim_var 或保留错误输入
                ylim_var.set("") # 清空无效输入
                ax.autoscale(enable=True, axis='y') # 恢复自动范围


            ax.set_xticks(x) 
            if any(len(str(label)) > 10 for label in plot_xticklabels):
                ax.set_xticklabels(plot_xticklabels, rotation=45, ha='right')
            else:
                ax.set_xticklabels(plot_xticklabels, rotation=0, ha='center')   
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            fig.tight_layout() # 使用 fig.tight_layout()
            canvas.draw()


        """title_var.set('Score from Log Files')
        xlabel_var.set('Log File')
        ylabel_var.set('Score Value')"""
        xtick_labels_var.set(str(log_file_names))


        # --- 绑定事件并进行初始绘图 ---
        metric_chooser.bind("<<ComboboxSelected>>", update_plot)
        
        # 初始绘图并填充 Entry 框
        update_plot()

        # 绑定窗口关闭事件
        plot_popup.protocol("WM_DELETE_WINDOW", lambda: self._on_plot_popup_close(plot_popup, fig))

    def _on_plot_popup_close(self, plot_popup, fig):
        """
        处理绘图弹窗关闭事件，释放 Matplotlib 资源。
        """
        plt.close(fig) # 关闭 Matplotlib 图，释放内存
        plot_popup.destroy() # 销毁 Tkinter 弹窗
    
        
        
        

# ===================================================================================================              
def return_task_method_list(task: str="tsp", file: str=UTILS_YAML) -> list:
    with open(file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    task_method_list = [key for key in config.keys() if config[key].get(task) is not None]  
    return task_method_list
    

def return_task_method_dict(task: str="tsp", method: str="pomo", file: str=UTILS_YAML) -> dict:  
    with open(file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file) 
    task_param_dict = config[method][task] 
    
    
    return task_param_dict


# ===================================================================================================
def flatten_dict(d, parent_key='', sep='.', exclude_keys=None, mode='test'):
    
    if exclude_keys is None:
        exclude_keys = set()
        
    if mode == 'test':
        exclude_keys.add('trainer')
    elif mode == 'train':
        exclude_keys.add('test_loader')
        
    items = []
    for k, v in d.items():
        if k in exclude_keys:
            continue  
       
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, str) and "${" in v:
            continue
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep, exclude_keys=exclude_keys))
        else:
            items.append((new_key, v))
    return items   




# ===================================================================================================



# ===================================================================================================


if __name__ == '__main__':

    app = MainApp()
    
    #app.configure(background = 'red')
    
    app.mainloop()