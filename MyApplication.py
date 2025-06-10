# UCG Data analyzer and modeling tool
# (c) 2020/2021
# Pre spustenie programu treba mať nainštalované:
# Python: https://www.python.org/downloads/
# Vývojové prostredie Pyzo: https://pyzo.org/start.html

"""
Institution: Technical University of Košice, Faculty BERG.
Department: Institute of Control and Informatization of Production Processes.
Website: www.tuke.sk
"""
# Doležité inštalácie (v python termináli):
# pip install image
# pip install matplotlib
# pip install sklearn
# alebo:
# pip install scikit-learn
# ...ak moduly nie su naištalované, tak python shell hlási chybu "ModuleNotFoundError"
# aktualizácia príkazu pip príkazovom riadku Win:  python -m pip install --upgrade pip

# Ako nainštalovať balíčky v IDE VSCode
# macOS:
# python3 -m pip install numpy

# Windows (may require elevation):
# py -m pip install numpy

# Linux (Debian):
# apt-get install python3-tk
# python3 -m pip install numpy


from PIL import Image, ImageTk   # Treba nainštalovať "pip install image"  !!!
from tkinter import Tk, Frame, Menu, Button, StringVar, Toplevel, Radiobutton, Checkbutton, Entry
from tkinter import LEFT, TOP, X, FLAT, RAISED
from tkinter import Label
from tkinter import LabelFrame
from tkinter.scrolledtext import ScrolledText
import os   # ...aby sme mohli pracovať s relatívnou cestou k súborom
from tkinter import filedialog
from matplotlib.figure import Figure # kreslenie grafov  # Treba nainštalovať "pip install matplotlib"  !!!
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from tkinter.messagebox import showerror
from pathlib import Path, PureWindowsPath
import csv
import sys
import numpy as np  # numerické operácie s maticami, skalármi a vektormi
from sklearn.preprocessing import StandardScaler  # ...pre štandardizáciu/normalizáciu dát treba importovať ale najprv naištalovať "pip install sklearn" !!!
from sklearn.svm import SVR # ...pre modelovanie metódou Support Vector Machines  t.j. regresia pomocou algoritmov strojového učenia
#import pandas as pd
#from tkinter import *
from tkinter import IntVar, StringVar  # potrebujeme pre radiobuttony, checkbuttony a editboxy


#RESOURCES_PATH = "F:\\PythonProjects"
RESOURCES_PATH = ".\\"

class MyApplication(Frame):


    def __init__(self):
        super().__init__()

        # Global variables
        self.time = [] # Time
        self.x1 = [] # Observation 1
        self.x2 = [] # Observation 2
        self.x3 = [] # Observation 3
        self.x4 = [] # Observation 4
        self.x5 = [] # Observation 5
        self.y = [] # Target 1

        self.observations_matrix = np.empty(shape=[0, 5]) # prázdna matica s piatimi stlpcami a neznamim počtom riadkov
        self.targets = np.empty(shape=[0, 1]) # prázdna matica/vektor s jedným stlpcom a neznamim počtom riadkov

        self.dataLoaded = False

        #self.editBoxVar= StringVar()
        self.editBoxVar= IntVar()
        self.radioButtonVar = IntVar()
        self.checkBoxVar1 = IntVar()
        self.checkBoxVar2 = IntVar()

        self.initUI()


    def initUI(self):

        self.master.title("Data analyzer and modeling tool")

        # *******************************************************
        # Vytvoríme hlavné menu aplikácie
        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        fileMenu = Menu(menubar)
        fileMenu.add_command(label="New")
        fileMenu.add_command(label="Open...", underline=0, command=self.onOpen) # # Položka menu s obsluženou udalosťou
        fileMenu.add_command(label="Save as...", underline=0, command=self.onSave) # # Položka menu s obsluženou udalosťou
        fileMenu.add_command(label="Close")
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", underline=0, command=self.onExit) # Položka menu s obsluženou udalosťou
        menubar.add_cascade(label="File", underline=0, menu=fileMenu)

        viewMenu = Menu(menubar)
        viewMenu.add_command(label="Show plots")
        viewMenu.add_command(label="Select data")
        viewMenu.add_command(label="Status bar")
        viewMenu.add_separator()
        viewMenu.add_command(label="Data zooming")
        menubar.add_cascade(label="View", underline=0, menu=viewMenu)

        settingsMenu = Menu(menubar)
        settingsMenu.add_command(label="Observations", underline=0, command=self.onObservations) # Položka menu s obsluženou udalosťou
        settingsMenu.add_command(label="Targets")
        settingsMenu.add_separator()
        settingsMenu.add_command(label="Preferencies", underline=0, command=self.onPreferencies) # Položka menu s obsluženou udalosťou
        settingsMenu.add_command(label="Modeling")
        menubar.add_cascade(label="Settings", underline=0, menu=settingsMenu)

        helpRun = Menu(menubar)
        helpRun.add_command(label="Prediction")
        helpRun.add_command(label="Analysis")
        helpRun.add_command(label="Data filtration")
        menubar.add_cascade(label="Run", underline=0, menu=helpRun)

        helpMenu = Menu(menubar)
        helpMenu.add_command(label="Website")
        helpMenu.add_command(label="Guide")
        helpMenu.add_separator()
        helpMenu.add_command(label="About this tool", underline=0, command=self.onAbout) # Položka menu s obsluženou udalosťou)
        menubar.add_cascade(label="Help", underline=0, menu=helpMenu)


        # *******************************************************
        # Načítame zdrojové obrázky pre toolbar alebo menu
        # automatické zistenie pristupovej cesty k nášmu skriptu (nemusí fungovať)
        # script_dir = os.path.dirname(__file__)
        script_dir = RESOURCES_PATH # pevná cesta

        rel_path_1 = "images\\new_2.png"
        abs_file_path_1 = os.path.join(script_dir, rel_path_1)
        self.img = Image.open(abs_file_path_1)
        eimg_1 = ImageTk.PhotoImage(self.img)

        rel_path_2 = "images\\open_2.png"
        abs_file_path_2 = os.path.join(script_dir, rel_path_2)
        self.img = Image.open(abs_file_path_2)
        eimg_2 = ImageTk.PhotoImage(self.img)

        rel_path_3 = "images\\save_2.png"
        abs_file_path_3 = os.path.join(script_dir, rel_path_3)
        self.img = Image.open(abs_file_path_3)
        eimg_3 = ImageTk.PhotoImage(self.img)

        rel_path_4 = "images\\modeling_settings_2.png"
        abs_file_path_4 = os.path.join(script_dir, rel_path_4)
        self.img = Image.open(abs_file_path_4)
        eimg_4 = ImageTk.PhotoImage(self.img)

        rel_path_5 = "images\\prediction_3.png"
        abs_file_path_5 = os.path.join(script_dir, rel_path_5)
        self.img = Image.open(abs_file_path_5)
        eimg_5 = ImageTk.PhotoImage(self.img)

        rel_path_6 = "images\\exit_2.png"
        abs_file_path_6 = os.path.join(script_dir, rel_path_6)
        self.img = Image.open(abs_file_path_6)
        eimg_6 = ImageTk.PhotoImage(self.img)

        # *******************************************************
        # Vytvoríme toolbar s dvoma tlačídlami a obrázkami
        toolbar = Frame(self.master, bd=1, relief=RAISED)

        # newButton = Button(toolbar, image=eimg_1, relief=FLAT,command=self.new)
        newButton = Button(toolbar, image=eimg_1, relief=FLAT)
        newButton.image = eimg_1
        newButton.pack(side=LEFT, padx=2, pady=2)

        newButton = Button(toolbar, image=eimg_2, relief=FLAT, command=self.onOpen) # Tlačidlo na toolbare s obsluženou udalosťou
        newButton.image = eimg_2
        newButton.pack(side=LEFT, padx=2, pady=2)

        newButton = Button(toolbar, image=eimg_3, relief=FLAT, command=self.onSave) # Tlačidlo na toolbare s obsluženou udalosťou
        newButton.image = eimg_3
        newButton.pack(side=LEFT, padx=2, pady=2)

        newButton = Button(toolbar, image=eimg_4, relief=FLAT, command=self.onPreferencies) # Tlačidlo na toolbare s obsluženou udalosťou
        newButton.image = eimg_4
        newButton.pack(side=LEFT, padx=2, pady=2)

        # newButton = Button(toolbar, image=eimg_5, relief=FLAT,command=self.modeling)
        newButton = Button(toolbar, image=eimg_5, relief=FLAT)
        newButton.image = eimg_5
        newButton.pack(side=LEFT, padx=2, pady=2)

        newButton = Button(toolbar, image=eimg_6, relief=FLAT, command=self.onExit) # Tlačidlo na toolbare s obsluženou udalosťou
        newButton.image = eimg_6
        newButton.pack(side=LEFT, padx=2, pady=2)

        toolbar.pack(side=TOP, fill=X)

        self.master.config(menu=menubar)

        # Vytvoríme stavový riadok
        self.statusBarText=StringVar()
        # vytvoríme stavový riadok z komponenty Label a priradíme mu reťazcovú premennú statusBarText
        #statusBar=Label(self.master, borderwidth=1, relief="sunken", textvariable=self.statusBarText)
        statusBar=Label(self.master, borderwidth=1, relief="sunken", textvariable=self.statusBarText, font=('arial',12,'normal'), anchor="w")
        statusBar.pack(side="bottom", fill="x")

        # *******************************************************
        # Vytvoríme päť pomenovaných skupín s objektami Frame v hlavnom okne pre výstup údajov

        # 1. rám:
        # Vytvoríme textové okno pre výpis správ programu
        self.group1 = LabelFrame(self.master, text="Message Window", padx=5, pady=5)
        self.group1.pack(side="bottom", fill="x")

        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(1, weight=1)

        self.group1.rowconfigure(0, weight=1)
        self.group1.columnconfigure(0, weight=1)

        # Textové výpisy budú realizované pomocou objektu typu ScrolledText
        self.group1.txtbox = ScrolledText(self.group1, width=40, height=10, bg='black', fg='white')
        self.group1.txtbox.grid(row=0, column=0, sticky='ewns')
        self.group1.txtbox.tag_config("here", background="black", foreground="green")
        self.group1.txtbox.insert('end',"Initializing... "+"\n")
        self.group1.txtbox.insert('end',"Creating base screen... "+"\n")
        self.group1.txtbox.insert('end',"Done. "+"\n")

        # 2. rám:
        self.group2 = LabelFrame(self.master, text="   Modeling adjustments   ", padx=5, pady=5)
        self.group2.pack(side="left", fill="y")
        self.group2.rowconfigure(0, weight=1)
        self.group2.columnconfigure(0, weight=1)
        self.group2.adjustmentsFrame = Frame(self.group2, width=20, height=40)
        self.group2.adjustmentsFrame.grid(row=0, column=0, sticky='ewns')
        # Na 2. rám vložíme nejaké widgety
        actionButton1 = Button(self.group2.adjustmentsFrame, text='Plot raw data', command=self.onPlotData).pack(fill="x")
        label1 = Label(self.group2.adjustmentsFrame, text="Test data percentage:").pack(fill="x")
        #self.editBoxVar.set("20")
        self.editBoxVar.set(20)
        edit1 = Entry(self.group2.adjustmentsFrame, textvariable=self.editBoxVar).pack(fill="x")
        actionButton2 = Button(self.group2.adjustmentsFrame, text='Prediction on test data', command=self.onPrediction).pack(fill="x")
        actionButton3 = Button(self.group2.adjustmentsFrame, text='Analysis/Training', command=self.onModelTraining).pack(fill="x")
        actionLabel1 = Label(self.group2.adjustmentsFrame,text='Kernel type').pack(fill="x")
        self.radioButtonVar.set(1)
        R1 = Radiobutton(self.group2.adjustmentsFrame, text="RBF/Gaussian", variable=self.radioButtonVar, value=1,  command=self.onRadioButtonClick).pack(fill="x")
        #R1.pack(anchor = 'w')
        R2 = Radiobutton(self.group2.adjustmentsFrame, text="Polynomial", variable=self.radioButtonVar, value=2, command=self.onRadioButtonClick).pack(fill="x")
        #R2.pack(anchor = 'w')
        R3 = Radiobutton(self.group2.adjustmentsFrame, text="Linear", variable=self.radioButtonVar, value=3, command=self.onRadioButtonClick).pack(fill="x")
        #R3.pack(anchor = 'w')
        actionButton4 = Button(self.group2.adjustmentsFrame, text='Refresh plots', command=self.onRefreshPlots).pack(fill="x")
        actionButton5 = Button(self.group2.adjustmentsFrame, text='Clear plots', command=self.onClearPlots).pack(fill="x")
        #actionButton6 = Button(self.group2.adjustmentsFrame, text='XXX').pack(fill="x")
        self.checkBoxVar1.set(1)
        self.checkBoxVar2.set(0)
        actionCheckBox1 = Checkbutton(self.group2.adjustmentsFrame, text='Method #1 (SVR)',variable=self.checkBoxVar1, onvalue=1, offvalue=0, command=self.onCheckButtonClick).pack(fill="x")
        actionCheckBox2 = Checkbutton(self.group2.adjustmentsFrame, text='Method #2 (XXX)',variable=self.checkBoxVar2, onvalue=1, offvalue=0, command=self.onCheckButtonClick).pack(fill="x")

        # 3. rám:
        self.group3 = LabelFrame(self.master, text="Plots", padx=5, pady=5)
        self.group3.pack(side="top", fill="both", expand=True)
        self.group3.rowconfigure(0, weight=1)
        self.group3.columnconfigure(0, weight=1)
        self.group3.plotsFrame = Frame(self.group3, width=10, height=40)
        self.group3.plotsFrame.grid(row=0, column=0, sticky='ewns')

        # V ráme #3 (plotsFrame) vytvoríme ďalšie dva vnorené rámy (jeden pre zobrazenie vtsupov a jeden pre zobrazenie výstupov)
        # 4. vnorený rám
        self.group3.plotsFrame.group4 = LabelFrame(self.group3.plotsFrame, text="Observations", padx=5, pady=5)
        self.group3.plotsFrame.group4.pack(side="top", fill="both", expand=True)
        self.group3.plotsFrame.group4.rowconfigure(0, weight=1)
        self.group3.plotsFrame.group4.columnconfigure(0, weight=1)
        self.group3.plotsFrame.group4.plotsObservationsFrame = Frame(self.group3.plotsFrame.group4, width=10, height=40)
        self.group3.plotsFrame.group4.plotsObservationsFrame.config(bg="white")
        self.group3.plotsFrame.group4.plotsObservationsFrame.grid(row=0, column=0, sticky='ewns')
        # 5. vnorený rám
        self.group3.plotsFrame.group5 = LabelFrame(self.group3.plotsFrame, text="Targets", padx=5, pady=5)
        self.group3.plotsFrame.group5.pack(side="top", fill="both", expand=True)
        self.group3.plotsFrame.group5.rowconfigure(0, weight=1)
        self.group3.plotsFrame.group5.columnconfigure(0, weight=1)
        self.group3.plotsFrame.group5.plotsTargetsFrame = Frame(self.group3.plotsFrame.group5, width=10, height=40)
        self.group3.plotsFrame.group5.plotsTargetsFrame.config(bg="white")
        self.group3.plotsFrame.group5.plotsTargetsFrame.grid(row=0, column=0, sticky='ewns')


        # Globalna kompresia/usporiadenie widgetov na self
        #self.pack()


    # *******************************************************
    #  Obsluha udalostí hlavného menu, toolbaru a tlačidliel

    #  Položka Menu->Open...
    def onOpen(self):
        file_name=filedialog.askopenfilename(filetypes=(("CSV files", "*.csv"),
        ("Text files", "*.txt"), ("Excel files", "*.xls;*.xlsx"), ("All files", "*.*") ))
        # Konverzia cesty k súboru na Windows formát
        #path_on_windows = PureWindowsPath(file_name)
        if file_name:
            try: # Názov súboru poznáme, tak vyskúšame súbor otvoriť a načítať

                # Otvoríme súbor
                csv_file = open(file_name,'rt')
                data = csv.reader(csv_file, delimiter=';')
                # Preskočíme prvý riadok v csv súbore
                next(data)

                # názov súboru poznáme, tak vyskúšame súbor otvoriť a načítať
                print("File Opened: "+file_name)
                self.statusBarText.set("File Opened: "+file_name)

                # Cyklické načítavanie riadkov
                self.group1.txtbox.insert('end',"Reading data from file..."+"\n")
                line_count = 0
                for row in data:
                    self.time.append(float(row[0]))    # Time
                    self.x1.append(float(row[1]))      # Observation 1
                    self.x2.append(float(row[2]))      # Observation 2
                    self.x3.append(float(row[3]))      # Observation 3
                    self.x4.append(float(row[4]))      # Observation 4
                    self.x5.append(float(row[5]))      # Observation 5
                    self.y.append(float(row[6]))       # Target 1

                    if line_count == 0:
                        # Nasledujúce dva riadky odblokuj ak chces výpisy do terminaloveho okna a/alebo okna správ programu
                        #print(f'Column names are {", ".join(row)}\n')
                        #self.group1.txtbox.insert('end',f'Column names are {", ".join(row)}\n')
                        line_count += 1
                    else:
                        # Nasledujúce dva riadky odblokuj ak chces výpisy do terminaloveho okna a/alebo okna správ programu
                        #print(f'\t Time: {row[0]} CO: {row[1]} CO2: {row[2]} H2: {row[3]} CH4: {row[4]} O2: {row[5]} Temp: {row[6]}\n')
                        #self.group1.txtbox.insert('end',f'\t Time: {row[0]} CO: {row[1]} CO2: {row[2]} H2: {row[3]} CH4: {row[4]} O2: {row[5]} Temp: {row[6]}\n')
                        line_count += 1

                print(f'Processed {line_count} lines.')
                self.group1.txtbox.insert('end',f'Processed {line_count} lines.\n')

                # Vytoríme maticu dát pozorovaní (observations) t.j. vstupov
                self.observations_matrix = np.array([self.x1,
                                                     self.x2,
                                                     self.x3,
                                                     self.x4,
                                                     self.x5]).T # Maticu transponujeme
                # Vytoríme vektor dát cieľových hodnôt (targets) t.j. výstupov
                self.targets = np.array([self.y]).T

                # Vypíšeme vstupno-výstupné dáta
                print("\n Observations´ matrice:")
                print(self.observations_matrix)
                print("\n Targets´ vector/matrice:")
                print(self.targets)

                print("File data loaded.")
                self.group1.txtbox.insert('end',"File data loaded successfully."+"\n")

                self.dataLoaded = True # Poznačíme, že dáta boli úspešne načítané


            except: # obsluženie výnimky
                showerror("Open Source File", "Failed to read file\n'%s'" %file_name)
                print("Failed to read file:\n "+file_name)
            finally:
                csv_file.close()

    #  Položka Menu->File->Save as...
    def onSave(self):
        file_name=filedialog.asksaveasfile(mode='w', defaultextension=".txt")
        if file_name is None: # asksaveasfile vráti `None` ak je dialogové okno zatvorené tlačidlom "cancel".
            return
        text2save = str(self.group1.txtbox.get(1.0, 'end')) # starts from `1.0`, not `0.0`
        file_name.write(text2save)
        file_name.close() # `()` was missing.
        print("File Saved.")
        self.group1.txtbox.insert('end',"File Saved. "+"\n")

    #  Položka Menu->File->Exit
    def onExit(self):
        self.master.destroy()

    # Položka Menu->Settings->Observations
    def onObservations(self):
        observationWindow = Toplevel(self)
        observationWindow.wm_title("Observations settings")
        l1 = Label(observationWindow, text="This window enable settings observation.s")
        l1.pack(side="top", fill="both", expand=True, padx=100, pady=100)
        b1 = Button(observationWindow, text='OK', command=observationWindow.destroy).pack()

    # Položka Menu->Settings->Preferencies
    def onPreferencies(self):
        preferenciesWindow = Toplevel(self)
        preferenciesWindow.wm_title("Preferencies")
        l1 = Label(preferenciesWindow, text="This window enable settings preferencies.")
        l1.pack(side="top", fill="both", expand=True, padx=100, pady=100)
        b1 = Button(preferenciesWindow, text='OK', command=preferenciesWindow.destroy).pack()

    # Položka Menu->Help->About
    def onAbout(self):
        aboutWindow = Toplevel(self)
        aboutWindow.wm_title("About this tool")
        l1 = Label(aboutWindow, text="This window provides info about this application.")
        l1.pack(side="top", fill="both", expand=True, padx=100, pady=100)
        b1 = Button(aboutWindow, text='OK', command=aboutWindow.destroy).pack()

    # Button "Plot data"
    def onPlotData(self):
        if self.dataLoaded == True:
            # Odstránime všetky staré widgety na ráme (ak nejaké sú)
            for widget in self.group3.plotsFrame.group4.plotsObservationsFrame.winfo_children():
                widget.destroy()
            # this will clear frame and frame will be empty
            # if you want to hide the empty panel then
            #frame.pack_forget()

            # Vytvoríme obrázok, ktorý bude obsahovať obrázok prvého grafu
            fig1 = Figure(figsize = (18, 2), dpi = 100)
            # adding the subplot
            plot1 = fig1.add_subplot(111)
            plot1.plot(self.time, self.x1, self.time, self.x2, self.time, self.x3, self.time, self.x4, self.time, self.x5)
            # creating the Tkinter canvas containing the Matplotlib figure
            canvas1 = FigureCanvasTkAgg(fig1, self.group3.plotsFrame.group4.plotsObservationsFrame)
            canvas1.draw()
            # placing the canvas on the Tkinter window
            canvas1.get_tk_widget().pack()

            # Odstránime všetky staré widgety na ráme (ak nejaké sú)
            for widget in self.group3.plotsFrame.group5.plotsTargetsFrame.winfo_children():
                widget.destroy()
            # this will clear frame and frame will be empty
            # if you want to hide the empty panel then
            #frame.pack_forget()

            # Vytvoríme obrázok, ktorý bude obsahovať obrázok druhého grafu
            fig2 = Figure(figsize = (18, 2), dpi = 100)
            # adding the subplot
            plot2 = fig2.add_subplot(111)
            # plotting the graph
            plot2.plot(self.time, self.y)
            # Vykresnenie na druhý frame
            canvas2 = FigureCanvasTkAgg(fig2, self.group3.plotsFrame.group5.plotsTargetsFrame)
            canvas2.draw()
            # placing the canvas on the Tkinter window
            canvas2.get_tk_widget().pack()

            print(f'Data have been displayed in plots.\n')
            self.group1.txtbox.insert('end',f'Data have been displayed in plots.\n')

        else:
            showerror("Plot data", "Failed to plot data. Please open the .csv file!")

    # Button "Refresh plots"
    def onRefreshPlots(self):
        if self.dataLoaded == True:
            # Odstránime všetky staré widgety na ráme (ak nejaké sú)
            for widget in self.group3.plotsFrame.group4.plotsObservationsFrame.winfo_children():
                widget.destroy()
            # this will clear frame and frame will be empty
            # if you want to hide the empty panel then
            #frame.pack_forget()

            # Vytvoríme obrázok, ktorý bude obsahovať obrázok prvého grafu
            fig1 = Figure(figsize = (18, 2), dpi = 100)
            # adding the subplot
            plot1 = fig1.add_subplot(111)
            plot1.plot(self.time, self.x1, self.time, self.x2, self.time, self.x3, self.time, self.x4, self.time, self.x5)
            # creating the Tkinter canvas containing the Matplotlib figure
            canvas1 = FigureCanvasTkAgg(fig1, self.group3.plotsFrame.group4.plotsObservationsFrame)
            canvas1.draw()
            # placing the canvas on the Tkinter window
            canvas1.get_tk_widget().pack()

            # Odstránime všetky staré widgety na ráme (ak nejaké sú)
            for widget in self.group3.plotsFrame.group5.plotsTargetsFrame.winfo_children():
                widget.destroy()
            # this will clear frame and frame will be empty
            # if you want to hide the empty panel then
            #frame.pack_forget()

            # Vytvoríme obrázok, ktorý bude obsahovať obrázok druhého grafu
            fig2 = Figure(figsize = (18, 2), dpi = 100)
            # adding the subplot
            plot2 = fig2.add_subplot(111)
            # plotting the graph
            plot2.plot(self.time, self.y)
            # Vykresnenie na druhý frame
            canvas2 = FigureCanvasTkAgg(fig2, self.group3.plotsFrame.group5.plotsTargetsFrame)
            canvas2.draw()
            # placing the canvas on the Tkinter window
            canvas2.get_tk_widget().pack()

            print(f'Plots have been refreshed.\n')
            self.group1.txtbox.insert('end',f'Plots have been refreshed.\n')

        else:
            showerror("Plot data", "Failed to plot data. Please open the .csv file!")

    # Kliknutie na RadioButton
    def onRadioButtonClick(self):
        if self.radioButtonVar.get() == 1:
            print("\nRadioButton #1 clicked.\n")
        if self.radioButtonVar.get() == 2:
            print("\nRadioButton #2 clicked.\n")
        if self.radioButtonVar.get() == 3:
            print("\nRadioButton #3 clicked.\n")

    # Kliknutie na CheckButton
    def onCheckButtonClick(self):
        if self.checkBoxVar1.get() == 1:
            print("CheckButton #1 clicked.\n")
        if self.checkBoxVar2.get() == 1:
            print("CheckButton #2 clicked.\n")

    # Kliknutie na tlačidlo "Clear plots"
    def onClearPlots(self):
        if self.dataLoaded == True:
            # Odstránime všetky staré widgety na ráme (ak nejaké sú)
            for widget in self.group3.plotsFrame.group4.plotsObservationsFrame.winfo_children():
                widget.destroy()
            # this will clear frame and frame will be empty
            # if you want to hide the empty panel then
            #frame.pack_forget()

            # Odstránime všetky staré widgety na ráme (ak nejaké sú)
            for widget in self.group3.plotsFrame.group5.plotsTargetsFrame.winfo_children():
                widget.destroy()
            # this will clear frame and frame will be empty
            # if you want to hide the empty panel then
            #frame.pack_forget()


    # Button "Analysis/Training"
    # Metóda natrénuje model na vsetkých načítaných dátach
    def onModelTraining(self):
        if self.dataLoaded == True:

            # Normalizácia dát (ak treba)
            #sc_X = StandardScaler()
            #sc_y = StandardScaler()

            X = self.observations_matrix
            y = self.targets

            #X = sc_X.fit_transform(self.observations_matrix)
            #y = sc_y.fit_transform(self.targets)

            # Nastavíme jadrovú funkciu pre model (Kernel type) podľa stavu RadioButtons
            if self.radioButtonVar.get() == 1:
                svr_model = SVR(kernel='rbf') # Gaussian regressor (Gaussian kernel)
            if self.radioButtonVar.get() == 2:
                svr_model = SVR(kernel='poly', C=1e3, degree=2)  # Polynomial regressor (Polynomial kernel)
            if self.radioButtonVar.get() == 3:
                svr_model = SVR(kernel='linear', C=1e3) # Linear regressor (Linear kernel)

            # Model natrénujeme na nameraných vstupoch a výstupoch
            self.group1.txtbox.insert('end',"Fitting/training SVR model.... "+"\n")
            svr_model.fit(X, y.ravel())
            self.group1.txtbox.insert('end',"The SVR model has been fitted. "+"\n")

            y_predicted = svr_model.predict(X)

            # zobrazíme výsledok predikcie
            print("\nMeasured data (y):\n")
            print(y)
            print("\nModel data (y_predicted):\n")
            print(y_predicted)




            # Odstránime všetky staré widgety na ráme (ak nejaké sú)
            for widget in self.group3.plotsFrame.group4.plotsObservationsFrame.winfo_children():
                widget.destroy()
            # this will clear frame and frame will be empty
            # if you want to hide the empty panel then
            #frame.pack_forget()

            # Vytvoríme obrázok, ktorý bude obsahovať obrázok prvého grafu
            fig1 = Figure(figsize = (18, 2), dpi = 100)
            # adding the subplot
            plot1 = fig1.add_subplot(111)
            plot1.plot(self.time, self.x1, self.time, self.x2, self.time, self.x3, self.time, self.x4, self.time, self.x5)
            #ax1 = plot1.plot(self.time, self.x1, self.time, self.x2, self.time, self.x3, self.time, self.x4, self.time, self.x5)
            #ax1.grid(True)
            #ax1.xaxis.set_xlabel('Time (s)')
            #ax1.set_xlabel('Time (s)')
            #ax1.set_ylabel('(%)')
            # creating the Tkinter canvas containing the Matplotlib figure
            canvas1 = FigureCanvasTkAgg(fig1, self.group3.plotsFrame.group4.plotsObservationsFrame)
            canvas1.draw()
            # placing the canvas on the Tkinter window
            canvas1.get_tk_widget().pack()

            # Odstránime všetky staré widgety na ráme (ak nejaké sú)
            for widget in self.group3.plotsFrame.group5.plotsTargetsFrame.winfo_children():
                widget.destroy()
            # this will clear frame and frame will be empty
            # if you want to hide the empty panel then
            #frame.pack_forget()

            # Vytvoríme obrázok, ktorý bude obsahovať obrázok druhého grafu
            fig2 = Figure(figsize = (18, 2), dpi = 100)
            # adding the subplot
            plot2 = fig2.add_subplot(111)
            # plotting the graph
            plot2.plot(self.time, self.y, self.time, y_predicted)
            #ax2 = plot2.plot(self.time, self.y, self.time, y_predicted)
            #ax2.grid(True)
            #ax2.xaxis.set_xlabel('Time (s)')
            #ax2.set_xlabel('Time (s)')
            #ax2.set_ylabel('Temperature (°C)')
            # Vykreslenie na druhý frame
            canvas2 = FigureCanvasTkAgg(fig2, self.group3.plotsFrame.group5.plotsTargetsFrame)
            canvas2.draw()
            # placing the canvas on the Tkinter window
            canvas2.get_tk_widget().pack()

            print(f'All observations and targets have been displayed in plots.\n')
            print(f'Model output has been displayed in plot.\n')
            self.group1.txtbox.insert('end',f'All observations and targets have been displayed in plots.\n')
            self.group1.txtbox.insert('end',f'Model output has been displayed in plot.\n')

            # Vypočítame MSE (Mean Squared Error)
            Sum = 0
            i=0
            for i in range(len(y_predicted)):
                Sum = Sum + ((self.y[i] - y_predicted[i]) ** 2)
            MSE = Sum / len(y_predicted)
            print(f'Mean Squared Error (MSE): {MSE} \n')
            self.group1.txtbox.insert('end',f'Mean Squared Error (MSE): {MSE} \n')

        else:
            showerror("Plot data", "Failed to plot data. Please open the .csv file!")

    # Metóda natrénuje model na vybranej trénovacej množine a vykoná predikciu výstupu na nenatrénovaných vstupoch
    def onPrediction(self):
        if self.dataLoaded == True:

            X = self.observations_matrix
            y = self.targets

            # Zistenie počtu vzoriek potrebných na trénovanie a na testovanie
            testDataPercentage = self.editBoxVar.get()
            totalSamplesCount = len(y)
            testSamplesCount = int((totalSamplesCount / 100) * testDataPercentage)
            trainSamplesCount = int(totalSamplesCount - testSamplesCount)
            print("\nSplit samples to train and test dataset:\n")
            self.group1.txtbox.insert('end',"\nSplit samples to train and test dataset:\n")
            print(f'Total samples count: {totalSamplesCount} \n')
            self.group1.txtbox.insert('end',f'Total samples count: {totalSamplesCount} \n')
            print(f'Train samples count: {trainSamplesCount} \n')
            self.group1.txtbox.insert('end',f'Train samples count: {trainSamplesCount} \n')
            print(f'Test  samples count: {testSamplesCount} \n')
            self.group1.txtbox.insert('end',f'Test  samples count: {testSamplesCount} \n')

            # Rozdelíme raw dáta (všetko čo bolo načítané zo súboru) na trénovaciu a testovaciu množinu.

            #X_train = [row[0:trainSamplesCount-1] for row in X]
            #X_test = [row[trainSamplesCount:totalSamplesCount-1] for row in X]
            #y_train = [row[0:trainSamplesCount-1] for row in y]
            #y_test = [row[trainSamplesCount:totalSamplesCount-1] for row in y]

            # Rozdelíme časové indície na dve časti t.j. pre trénovaciu a pre testovaciu množinu dát
            time_train = []
            time_test = []
            line_count = 1
            for row in self.time:
                if line_count <= trainSamplesCount:
                    time_train.append(row)
                if line_count > trainSamplesCount:
                    time_test.append(row)
                line_count += 1

            # Vytvoríme trénovacie a testovacie vstupy
            X_train = []
            X_test = []
            line_count = 1
            for row in X:
                if line_count <= trainSamplesCount:
                    X_train.append(row)
                if line_count > trainSamplesCount:
                    X_test.append(row)
                line_count += 1

            # Vytvoríme trénovacie a testovacie výstupy
            y_train = []
            y_test = []
            line_count = 1
            for row in y:
                if line_count <= trainSamplesCount:
                    y_train.append(row)
                if line_count > trainSamplesCount:
                    y_test.append(row)
                line_count += 1

            row = len(time_train)
            col = 1
            print(f'Time indicies dimension for train input-dataset: {row}x{col} (observations time matrice) \n')
            self.group1.txtbox.insert('end',f'Time indicies dimension for train input-dataset: {row}x{col} (observations time matrice) \n')
            row = len(time_test)
            col = 1
            print(f'Time indicies dimension for test input-dataset: {row}x{col} (observations time matrice) \n')
            self.group1.txtbox.insert('end',f'Time indicies dimension for test input-dataset: {row}x{col} (observations time matrice) \n')

            row = len(X_train)
            col = len(X_train[0])
            print(f'Train input-dataset dimesion: {row}x{col} (observations matrice) \n')
            self.group1.txtbox.insert('end',f'Train input-dataset dimesion: {row}x{col} (observations matrice) \n')
            row = len(X_test)
            col = len(X_test[0])
            print(f'Test input-dataset dimesion: {row}x{col} (observations matrice)\n')
            self.group1.txtbox.insert('end',f'Test input-dataset dimesion: {row}x{col} (observations matrice)\n')
            row = len(y_train)
            col = len(y_train[0])
            print(f'Train output-dataset dimesion: {row}x{col} (targets matrice/vector) \n')
            self.group1.txtbox.insert('end',f'Train output-dataset dimesion: {row}x{col} (targets matrice/vector) \n')
            row = len(y_test)
            col = len(y_test[0])
            print(f'Test output-dataset dimesion: {row}x{col} (targets matrice/vector)\n')
            self.group1.txtbox.insert('end',f'Test output-dataset dimesion: {row}x{col} (targets matrice/vector)\n')

            # Nastavíme jadrovú funkciu pre model (Kernel type) podľa stavu RadioButtons
            if self.radioButtonVar.get() == 1:
                svr_model = SVR(kernel='rbf') # Gaussian regressor (Gaussian kernel)
            if self.radioButtonVar.get() == 2:
                svr_model = SVR(kernel='poly', C=1e3, degree=2)  # Polynomial regressor (Polynomial kernel)
            if self.radioButtonVar.get() == 3:
                svr_model = SVR(kernel='linear', C=1e3) # Linear regressor (Linear kernel)

            # Model natrénujeme na nameraných vstupoch a výstupoch
            self.group1.txtbox.insert('end',"Fitting/training SVR model.... "+"\n")
            svr_model.fit(X_train, np.array(y_train).ravel())
            self.group1.txtbox.insert('end',"The SVR model has been fitted. "+"\n")


            y_predicted_train = svr_model.predict(X_train) # predikcia s natrénovaným SVR modelom na natrénovaných dátach (vstupoch)
            y_predicted_test = svr_model.predict(X_test)  # predikcia s natrénovaným SVR modelom na nenatrénovaných dátach (vstupoch)

            # Zobrazíme výsledok predikcie z trénovacej fázy t.j. ako model aproximuje merané dáta y
            print("\nMeasured and prediced target from the training phase:\n")
            print("\nMeasured data (y_train):\n")
            print(y_train)  # merané dáta
            print("\nModel data (y_predicted_train):\n")
            print(y_predicted_train)  # modelové, predikované dáta

            # Zobrazíme výsledok predikcie z testovacej fázy t.j. ako model predikuje y z nenatrénovaných dát (vstupov/pozorovaní)
            print("\nMeasured and prediced target from the testing phase:\n")
            print("\nMeasured data (y_test):\n")
            print(y_train)  # merané dáta
            print("\nModel data (y_predicted_test):\n")
            print(y_predicted_train)  # modelové, predikované dáta

            # Výpočet MSE pre trénovaciu fázu a pre testovaciu fázu:
            # Vypočítame MSE (Mean Squared Error) pre trénovaciu fázu
            Sum = 0
            i=0
            for i in range(len(y_predicted_train)):
                Sum = Sum + ((y_train[i] - y_predicted_train[i]) ** 2)
            MSE = Sum / len(y_predicted_train)
            print(f'Mean Squared Error (MSE) in train phase: {MSE} \n')
            self.group1.txtbox.insert('end',f'Mean Squared Error (MSE) in train phase: {MSE} \n')

            # Vypočítame MSE (Mean Squared Error) pre testovaciu fázu
            Sum = 0
            i=0
            for i in range(len(y_predicted_test)):
                Sum = Sum + ((y_test[i] - y_predicted_test[i]) ** 2)
            MSE = Sum / len(y_predicted_test)
            print(f'Mean Squared Error (MSE) in test phase: {MSE} \n')
            self.group1.txtbox.insert('end',f'Mean Squared Error (MSE) in test phase: {MSE} \n')

            # Vykreslenie dát  (len testovacie namerané pozorovania, namerané cieľe a predikované ciele):

            # Odstránime všetky staré widgety na ráme (ak nejaké sú)
            for widget in self.group3.plotsFrame.group4.plotsObservationsFrame.winfo_children():
                widget.destroy()
            # this will clear frame and frame will be empty
            # if you want to hide the empty panel then
            #frame.pack_forget()

            # Vytvoríme obrázok, ktorý bude obsahovať obrázok prvého grafu
            fig1 = Figure(figsize = (18, 2), dpi = 100)
            # adding the subplot
            plot1 = fig1.add_subplot(111)

            # Vykreslíme pozorovania (vstupy) z testovacej dátovej množiny
            plot1.plot(time_test, np.array(X_test)[:,0], time_test, np.array(X_test)[:,1], time_test, np.array(X_test)[:,2], time_test, np.array(X_test)[:,3], time_test, np.array(X_test)[:,4])

            # creating the Tkinter canvas containing the Matplotlib figure
            canvas1 = FigureCanvasTkAgg(fig1, self.group3.plotsFrame.group4.plotsObservationsFrame)
            canvas1.draw()
            # placing the canvas on the Tkinter window
            canvas1.get_tk_widget().pack()

            # Odstránime všetky staré widgety na ráme (ak nejaké sú)
            for widget in self.group3.plotsFrame.group5.plotsTargetsFrame.winfo_children():
                widget.destroy()
            # this will clear frame and frame will be empty
            # if you want to hide the empty panel then
            #frame.pack_forget()

            # Vytvoríme obrázok, ktorý bude obsahovať obrázok druhého grafu
            fig2 = Figure(figsize = (18, 2), dpi = 100)
            # adding the subplot
            plot2 = fig2.add_subplot(111)
            # plotting the graph
            # Vykreslíme ciele (merané a pridikované výstupy) z testovacej dátovej množiny
            plot2.plot(time_test, y_test, time_test, y_predicted_test)

            # Vykresnenie na druhý frame
            canvas2 = FigureCanvasTkAgg(fig2, self.group3.plotsFrame.group5.plotsTargetsFrame)
            canvas2.draw()
            # placing the canvas on the Tkinter window
            canvas2.get_tk_widget().pack()


            print(f'All observations and targets from testing phase have been displayed in plots.\n')
            print(f'Model output from testing phase has been displayed in plot.\n')
            self.group1.txtbox.insert('end',f'All observations and targets from testing phase have been displayed in plots.\n')
            self.group1.txtbox.insert('end',f'Model output from testing phase has been displayed in plot.\n')


        else:
            showerror("Plot data", "Failed to plot data. Please open the .csv file!")



    #    return


def main():

    root = Tk()
    root.geometry("800x800+150+150")
    app = MyApplication()
    root.mainloop()

if __name__ == '__main__':
    main()

# Zdroje:
# https://stackoverflow.com/questions/41906206/pip-unicodedecodeerror-utf8-codec-cant-decode-byte
# https://www.geeksforgeeks.org/python-pack-method-in-tkinter/
# https://pythonprogramming.net/how-to-embed-matplotlib-graph-tkinter-gui/
# https://swcarpentry.github.io/python-novice-gapminder/09-plotting/
# https://datatofish.com/matplotlib-charts-tkinter-gui/
# https://pythonspot.com/tk-menubar/
# http://zetcode.com/tkinter/menustoolbars/
# https://pythonprogramming.altervista.org/a-toolbar-for-python-with-tkinter/
# https://www.tutorialspoint.com/python/tk_button.htm
# https://stackoverflow.com/questions/15306631/how-do-i-create-child-windows-with-python-tkinter
# https://stackoverflow.com/questions/8120246/python-tkinter-program-for-status-bar
# https://pythonspot.com/tk-file-dialogs/
# https://stackoverflow.com/questions/9239514/filedialog-tkinter-and-opening-files
# https://gist.github.com/Yagisanatode/0d1baad4e3a871587ab1
# https://pandas.pydata.org/getting_started.html
# https://stackoverflow.com/questions/5459444/tkinter-python-may-not-be-configured-for-tk
# https://stackoverflow.com/questions/20044559/how-to-pip-or-easy-install-tkinter-on-windows
# https://stackabuse.com/python-gui-development-with-tkinter-part-2/
# https://pythonprogramming.net/loading-file-data-matplotlib-tutorial/
# https://stackoverflow.com/questions/38532298/how-can-you-plot-data-from-a-txt-file-using-matplotlib
# https://stackoverflow.com/questions/30690365/python-3-tkinter-button-command-inside-different-class
# https://pythonprogramming.net/how-to-embed-matplotlib-graph-tkinter-gui/
# https://stackoverflow.com/questions/42514506/tkinter-binding-to-inside-a-function
# https://subscription.packtpub.com/book/application_development/9781788837460/1/ch01lvl1sec17/events-and-callbacks-adding-life-to-programs
# https://stackoverflow.com/questions/44464686/opening-file-path-not-working-in-python
# https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
# https://stackoverflow.com/questions/15781802/python-tkinter-clearing-a-frame
# https://stackoverflow.com/questions/22276066/how-to-plot-multiple-functions-on-the-same-figure-in-matplotlib
# a iné...
# Machine Learning:
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
# https://calebshortt.com/2016/01/15/installing-scikit-learn-python-data-mining-library/
# https://medium.com/pursuitnotes/support-vector-regression-in-6-steps-with-python-c4569acd062d
# https://www.oreilly.com/library/view/machine-learning-with/9781491989371/ch01.html
# https://machinelearningmastery.com/multivariate-adaptive-regression-splines-mars-in-python/
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
# https://www.datacamp.com/community/tutorials/pandas-read-csv
# https://www.educative.io/courses/data-analysis-processing-with-pandas/q2pYRjnxlGR
# https://docs.python.org/3/tutorial/inputoutput.html
# https://stackoverflow.com/questions/568962/how-do-i-create-an-empty-array-matrix-in-numpy
# https://stackoverflow.com/questions/21008858/formatting-floats-in-a-numpy-array
# https://www.oreilly.com/library/view/machine-learning-with/9781491989371/ch01.html
# https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
# https://stackoverflow.com/questions/27372105/tkinter-radiobutton-intvar-attribute-error
# https://stackoverflow.com/questions/32729051/python-progress-bar-with-dynamic-length
# http://zetcode.com/articles/tkinterlongruntask/
# http://zetcode.com/all/#python
# https://www.root.cz/clanky/spracovanie-dlhotrvajucej-ulohy-v-tkinteri/
# https://en.wikipedia.org/wiki/Mean_squared_error
# https://machinelearningmastery.com/multivariate-adaptive-regression-splines-mars-in-python/
# https://stackoverflow.com/questions/16373887/how-to-set-the-text-value-content-of-an-entry-widget-using-a-button-in-tkinter/54040005
# https://stackoverflow.com/questions/903853/how-do-you-extract-a-column-from-a-multi-dimensional-array
# https://stackoverflow.com/questions/38662667/matplotlib-and-subplots-properties
