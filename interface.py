import tkinter as tk
from practicalWork2_avecPenteEtFrottement import launch
from interpreteurtxt import Param,push_settings


def go():

    push_settings(Param, "settings.txt")
    launch()

def execution():
    """
  Lance la fenetre principale
  -----
  entrée :
  Rien
  -----
  retour :
  Rien
  """
    # Configuration
    window = tk.Tk()  # commande de creation d'application
    window.title("Solver")  # titre de l'application
    window.geometry("500x90")  # taille de la fenetre
    window.config(background='#42B79F')  # couleur de fond de la fenetre
    window.resizable(width=False, height=False)  # la fenetre a une taille fixe
    # Configuration

    ##### bouton lancer #####
    bpushaxe = tk.Button(window, text="LANCER LE CALCUL", command=go, bg='#72aaab', width=68, height=4)
    bpushaxe.place(x=5, y=10)
    ##### bouton lancer #####
    var_flux = tk.IntVar()
    if Param["flux"] == "Naif":
        var_flux.set(1)
    if Param["flux"] == "LaxFr":
        var_flux.set(2)
    if Param["flux"] == "Rusanov":
        var_flux.set(3)

    def choix_flux():
        if var_flux.get() == 1:
            Param["flux"] = "Naif"
        if var_flux.get() ==2:
            Param["flux"] = "LaxFr"
        if var_flux.get() ==3:
            Param["flux"] = "Rusanov"

    menubar = tk.Menu(window)
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_radiobutton(label="Naif", command=choix_flux, variable=var_flux, value=1)
    filemenu.add_radiobutton(label="Lax Friedrich", command=choix_flux, variable=var_flux, value=2)
    filemenu.add_radiobutton(label="Rusanov", command=choix_flux, variable=var_flux, value=3)
    menubar.add_cascade(label="Flux", menu=filemenu)

    var_ci = tk.IntVar()
    if Param["cond_ini"] == "Perturb":
        var_ci.set(1)
    if Param["cond_ini"] == "Dam":
        var_ci.set(2)
    if Param["cond_ini"] == "Flat":
        var_ci.set(3)
    if Param["cond_ini"] == "Step":
        var_ci.set(4)

    def choix_ci():
        if var_ci.get() == 1:
            Param["cond_ini"] = "Perturb"
        if var_ci.get() == 2:
            Param["cond_ini"] = "Dam"
        if var_ci.get() == 3:
            Param["cond_ini"] = "Flat"
        if var_ci.get() == 4:
            Param["cond_ini"] = "Step"
    editmenu = tk.Menu(menubar, tearoff=0)
    editmenu.add_radiobutton(label="Perturbation", command=choix_ci, variable=var_ci, value=1)
    editmenu.add_radiobutton(label="Rupture de barrage", command=choix_ci, variable=var_ci, value=2)
    editmenu.add_radiobutton(label="Plat", command=choix_ci, variable=var_ci, value=3)
    editmenu.add_radiobutton(label="Echellon", command=choix_ci, variable=var_ci, value=4)
    menubar.add_cascade(label="Condition initiale", menu=editmenu)

    var_cl = tk.IntVar()
    if Param["cond_bound"] == "Period":
        var_cl.set(1)
    if Param["cond_bound"] == "Dirich":
        var_cl.set(2)
    if Param["cond_bound"] == "Neum":
        var_cl.set(3)
    if Param["cond_bound"] == "Mixed":
        var_cl.set(4)

    def choix_ci():
        if var_cl.get() == 1:
            Param["cond_bound"] = "Period"
        if var_cl.get() == 2:
            Param["cond_bound"] = "Dirich"
        if var_cl.get() == 3:
            Param["cond_bound"] = "Neum"
        if var_cl.get() == 4:
            Param["cond_bound"] = "Mixed"
    helpmenu = tk.Menu(menubar, tearoff=0)
    helpmenu.add_radiobutton(label="Périodique",  command=choix_ci, variable=var_cl, value=1)
    helpmenu.add_radiobutton(label="Dirichlet",  command=choix_ci, variable=var_cl, value=2)
    helpmenu.add_radiobutton(label="Neumann",  command=choix_ci, variable=var_cl, value=3)
    helpmenu.add_radiobutton(label="Mixte",  command=choix_ci, variable=var_cl, value=4)
    menubar.add_cascade(label="Condition limite", menu=helpmenu)

    pente = tk.IntVar()

    if Param["slope"]:
        pente.set(1)
    else:
        pente.set(0)
    frict = tk.IntVar()

    if Param["frict"]:
        frict.set(1)
    else:
        frict.set(0)

    def choix_pente():
        Param["slope"] = not Param["slope"]

    def choix_frict():
        Param["frict"] = not Param["frict"]

    pentemenu = tk.Menu(menubar, tearoff=0)
    pentemenu.add_radiobutton(label="Pente",  command=choix_pente,  variable=pente, value=1)
    pentemenu.add_radiobutton(label="Frottements",  command=choix_frict, variable=frict, value=1)
    menubar.add_cascade(label="Termes de charge", menu=pentemenu)

    nummenu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="constantes", menu=nummenu)

    window.config(menu=menubar)
    window.mainloop()
