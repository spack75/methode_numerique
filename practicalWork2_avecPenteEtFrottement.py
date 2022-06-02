# Code to solve numerically an hyperbolic equation (dh/dt + lambda dh/dx = 0) with finite difference method
# Solve numerically the case of a perturbation with forward-time backward-space numerical scheme and periodic boundary
# conditions
# Plot the results as an animated plot over time
# Written by Raphael MAURIN (raphael.maurin@imft.fr) after a document of Pierre-Yves LAGREE: Resolution numerique des
# equations de Saint-Venant, mise en oeuvre en volumes finis par un solveur de Riemann bien balance
# 10/05/2022

####################################
# READ ME
# To make the script work, you should prescribed an option to True for the Flux you want to use (naive, Lax-Friedrich,
# or Rusanov) and for the configuration you want to consider (dam break or perturbation)

# Structure of the script:
# 1. Options & parameters of the simulation (to set)
# 2. Prescribe initial configuration
# 3. Loop over time for the equation resolution
# ... 3.1 Evaluate the flux function at every half spatial step xi-1/2
# ... 3.2 Evaluate h^{n+1} and q^{n+1} from the flux function
# ... 3.3 Set the boundary conditions for the next time step
# 4. Animated plot of the results (h,q) as a function of time

# Import useful libraries

import numpy as np  # To deal with vector and matrix
import matplotlib.pyplot as plt  # To plot figures
from math import *  # Import mathematic library
import matplotlib.animation as animation  # Import libraries to make an animated plot
from matplotlib.lines import Line2D  # Import libraries to make an animated plot
from interpreteurtxt import Param
############################################################################################################
# 1. Options & parameters of the simulation
############################################################################################################
slopeTermCalculation = Param["slope"]
frictionTermCalculation = Param["frict"]
Ks = 50.  # Strickler coefficient

# _____________ TO SET _____________
# Options for the flux for the numerical resolution, see what it modifies in the temporal loop below
naiveFlux = (Param["flux"] == "Naif")  # Naive formulation of the flux (unstable) Xflat Xstep Xperturbation Xdam
LaxFriedrichFlux = (Param["flux"] == "LaxFr")  # Lax-Friedrich formulation of the flux (stable but diffusion)
RusanovFlux = (Param["flux"] == "Rusanov")  # Rusanov formulation of the flux

# _____________ TO SET _____________
# Options for the initial condition
Perturbation = (Param["cond_ini"] == "Perturb") # a corriger
DamBreak = (Param["cond_ini"] == "Dam") # a corriger
Flat = (Param["cond_ini"] == "Flat")
Step = (Param["cond_ini"] == "Step")

# _____________ TO SET_____________
# Options for the boundary conditions
periodicBC = (Param["cond_bound"] == "Period")  # Periodic boundary conditions, for both h and q
dirichletBC = (Param["cond_bound"] == "Dirich")  # Dirichlet boundary conditions, for both h and q, on both sides
neumannBC = (Param["cond_bound"] == "Neum")  # Neumann boundary conditions, for both h and q, on both sides
mixedBC = (Param["cond_bound"] == "Mixed")  # Mixed boundary conditions, to set by the user at the end of section 3

########################################################################
# CHARACTERISTICS OF THE EQUATIONS AND OF THE NUMERICAL RESOLUTION

# Size of the domain and time simulated
tSimulated = 300.  # Simulated time, in s
length = 1000.  # Simulated length, in m
# Spatial and temporal resolution
dt = 1e-1  # Time step, in s
dx = 1.  # Grid size, in m
# Gravity
g = 9.81

########################################################################
# Initialization of the (time & space) discretized vector h

# Size of the spatio-temporal domain and initialization
Nt = int(tSimulated / dt)  # Number of time step
Nx = int(length / dx)  # Number of grid element
h = np.zeros(
    [Nt, Nx + 2])  # Initialize the water depth matrix to 0, h[t,x] = water depth at time step t and grid point x
q = np.zeros(
    [Nt, Nx + 2])  # Initialize the water discharge matrix to 0,q[t,x]=water depth at timestep t and grid point x
X = np.linspace(0, length, Nx)  # Define the spatial mesh
slopeTerm = 0.
frictionTerm = 0.

########################################################################
# Prescribe the bottom
Z = np.zeros(Nx + 2)
Z[:int(Nx / 2) + 1] = np.linspace(1.5, 0., int(Nx / 2) + 1)
Z = np.linspace(1., 0., Nx + 2)

########################################################################
# Test on the options, to be sure that only one has been set (not important)
if (naiveFlux and LaxFriedrichFlux) or (RusanovFlux and LaxFriedrichFlux) or (naiveFlux and RusanovFlux):
    print('There is a problem, too many options have been set for the flux\n')
    exit()
elif not naiveFlux and not LaxFriedrichFlux and not RusanovFlux:
    print('There is a problem, no option has been set for the flux\n')
    exit()
if Perturbation and DamBreak:
    print('There is a problem, two options have been given for the initial conditions\n')
    exit()
elif not Perturbation and not DamBreak and not Flat and not Step:
    print('There is a problem, no option has been given for the initial conditions\n')
    exit()
if DamBreak and naiveFlux:  # For a dam break problem, the naive flux formulation leads to a very quick instability,
    # restrict the number of time step then.
    nt = 60

############################################################################################################
# 2. Prescribe initial configuration
############################################################################################################

href = 1.  # Typical water depth considered
# qref = 0.  # Reference water discharge (per unit width)

# Impose a Fr number and a given according discharge
Fr = 1.2  # Fr = U/sqrt(gh) = q/sqrt(gh^3) --> q = Fr sqrt(gh^3)
qref = Fr * sqrt(g * href ** 3)

if Perturbation:
    # Take it as a gaussian perturbation over the equilibrium state hn for example 
    # (we could imagine any initial condition, depending on the configuration/problem we study)
    A = href / 5.  # Amplitude of the perturbation
    x_a = length / 2.  # Center of the gaussian perturbation
    sig = length / 30.  # width of the perturbation
    for nx in range(0, Nx + 2):
        x = nx * dx
        h[0, nx] = max(0., href + A * exp(-pow(x - x_a, 2) / (2. * pow(sig, 2))) - Z[nx])
if DamBreak:
    h[0, :int(Nx / 2)] = href
    h[0, int(Nx / 2):] = href/2.
    q[0, :] = qref
if Flat:
    # Lake at rest
    # for nx in range(0,Nx+2):
    # h[0,nx] = max(href-Z[nx],0.)
    # qref = 0.
    # Flat with a given discharge
    h[0, :] = href
    q[0, :] = qref

if Step:
    Fr_l = 1.2  # Fr = U/sqrt(gh) = q/sqrt(gh^3) --> h = (q^2/Fr^2/)^1/3
    Fr_r = 0.8
    hl = pow(qref ** 2 / Fr_l ** 2 / g, 1 / 3.)
    hr = pow(qref ** 2 / Fr_r ** 2 / g, 1 / 3.)
    h[0, :int(Nx / 2)] = hl
    h[0, int(Nx / 2):] = hr
    q[0, :] = qref

############################################################################################################
# 3. Loop over time for the equation resolution
############################################################################################################

for nt in range(0, Nt - 1):  # Loop over time
    if int(nt/Nt*100) < int((nt+1)/Nt*100):
        print(f"calcul {int((nt+1)/Nt*100)}% completed")
    # Condition to avoid instability
    if h[nt, :].sum() == 0:
        print('\n !!!! No more water depth, stop the calculation !!!')
        break
    if h[nt, :].min() < 0:
        print('\n !!!! Negative water depth, stop the calculation at time step: ', nt, "!!!\n")
        break

    ####################################################################################################
    # 3.1 Evaluate the flux function at every half spatial step xi-1/2

    # Initialize the fluxes
    FMass = np.zeros(Nx + 2)
    FMom = np.zeros(Nx + 2)
    c = 0.
    # Loop over space to calculate the flux
    for nx in range(1, Nx + 2):
        # Calculate the flux for Naive, Lax Friedrich or Rusanov formulation
        # Evaluation of the celerity
        if naiveFlux:
            # Naive 
            c = 0.
        if LaxFriedrichFlux:
            # Lax-Friedrich
            c = dx / dt
        if RusanovFlux:
            # Rusanov 
            if h[nt, nx] > 0.01:
                cnx = abs(q[nt, nx] / h[nt, nx]) + sqrt(g * h[nt, nx])
            else:
                cnx = 0.
            if h[nt, nx - 1] > 0.01:
                cnxm = abs(q[nt, nx - 1] / h[nt, nx - 1]) + sqrt(g * h[nt, nx - 1])
            else:
                cnxm = 0.
            c = max(cnx, cnxm)
        #print("c", c)
        # Calculation of the flux
        FMass[nx] = (q[nt, nx] + q[nt, nx - 1]) / 2. - c * (h[nt, nx] - h[nt, nx - 1]) / 2.
        if h[nt, nx] > 0 and h[nt, nx - 1] > 0:
            FMom[nx] = (q[nt, nx] ** 2 / h[nt, nx] + 0.5 * g * h[nt, nx] ** 2 + q[nt, nx - 1] ** 2 / h[nt, nx - 1] +
                        0.5 * g * h[nt, nx - 1] ** 2) / 2. - c * (q[nt, nx] - q[nt, nx - 1]) / 2.
        elif h[nt, nx] <= 0 and h[nt, nx - 1] <= 0:
            FMom[nx] = 0.
        elif h[nt, nx] > 0 >= h[nt, nx - 1]:
            FMom[nx] = (q[nt, nx] ** 2 / h[nt, nx] + 0.5 * g * h[nt, nx] ** 2) / 2. - c * (
                        q[nt, nx] - q[nt, nx - 1]) / 2.
        elif h[nt, nx] <= 0 < h[nt, nx - 1]:
            FMom[nx] = (q[nt, nx - 1] ** 2 / h[nt, nx - 1] + 0.5 * g * h[nt, nx - 1] ** 2) / 2. - c * (
                        q[nt, nx] - q[nt, nx - 1]) / 2.
    ####################################################################################################
    # 3.2 Evaluate h^{n+1} and q^{n+1} from the flux function

    # Loop over space to solve the height and discharge at next time step
    for nx in range(1, Nx + 1):
        # Evaluate the slope term if activated
        if slopeTermCalculation:
            hig = max(0., h[nt, nx - 1] + Z[nx - 1] - max(Z[nx - 1], Z[nx]))
            hid = max(0., h[nt, nx] + Z[nx] - max(Z[nx - 1], Z[nx]))
            slopeTerm = - g / 2. * (hig ** 2 - h[nt, nx - 1] ** 2 + h[nt, nx] ** 2 - hid ** 2) / dx
        # Evaluate the friction term if activated
        if frictionTermCalculation:
            if h[nt, nx] > 1e-3:
                frictionTerm = g / Ks ** 2 / h[nt, nx]**(7./3.) * abs(q[nt, nx]) * q[nt, nx]
            else:
                frictionTerm = 0.

        # Calculate h and q at next step from the contribution of the advection, the slope term and the friction term.
       # print(f"t {nt} \n x {nx} \nFmass {FMass[nx]} \n Fmom{FMom[nx]} \n q {q[nt, nx]} \n h {h[nt, nx]}")
        h[nt + 1, nx] = max(h[nt, nx] - dt / dx * (FMass[nx + 1] - FMass[nx]), 0.)
        if h[nt + 1, nx] > 0:
            q[nt + 1, nx] = q[nt, nx] - dt / dx * (FMom[nx + 1] - FMom[nx]) - dt * slopeTerm - dt * frictionTerm
        else:
            q[nt + 1, nx] = 0.

    ####################################################################################################
    #  3.3 Set the boundary conditions for the next time step
    # Apply boundary conditions in the two ghost cells (nx = 0 and nx = Nx+1) for q and h

    if periodicBC:
        h[nt + 1, 0] = h[nt + 1, 1]
        h[nt + 1, Nx + 1] = h[nt + 1, Nx]
        q[nt + 1, 0] = q[nt + 1, 1]
        q[nt + 1, Nx + 1] = q[nt + 1, Nx]
    if dirichletBC:
        h[nt + 1, 0] = 1.1 * href
        h[nt + 1, Nx + 1] = href
        q[nt + 1, 0] = qref
        q[nt + 1, Nx + 1] = qref
    if neumannBC:
        h[nt + 1, 0] = h[nt + 1, 1]
        h[nt + 1, Nx + 1] = h[nt + 1, Nx]
        q[nt + 1, 0] = q[nt + 1, 1]
        q[nt + 1, Nx + 1] = q[nt + 1, Nx]
    if mixedBC:
        # Impose the same discharge along the channel
        q[nt + 1, 0] = qref
        q[nt + 1, Nx + 1] = qref

        # Let h vary at the entry of the channel
        h[nt + 1, 0] = h[nt + 1, 1]

        # Put a gate at the right, has to be Fr = 1
        Fr_r = 1.
        h[nt + 1, Nx + 1] = pow(qref ** 2 / Fr_r ** 2 / g,
                                1 / 3.)  # Fr = U/sqrt(gh) = q/sqrt(gh^3) --> h = (q^2/Fr^2/)^1/3

################################################################################
# 4. Animated plot of the results (h,q) as a function of time
################################################################################

fig = plt.figure(figsize=(12, 10))  # Create a figure
ax1 = fig.add_subplot(2, 1, 1)  # divide the figure in 1 subplot

line = Line2D([], [], color='b')  # create a line of color blue, that will be used for the anomaed plot
ax1.add_line(line)  # Add the corresponding line to the subplot
ax1.set_xlabel('x (in m)')  # Label the x axis
ax1.set_ylabel('h (in m)')  # Label the y axis
ax1.set_xlim([0, length])  # Fix the length of the x axis we will see
ax1.set_ylim([0, np.amax(Z) + 1.1 * np.max(h)])  # Fix the length of the x axis we will see
time_text1 = ax1.text(0.1, 0.1, '', transform=ax1.transAxes)

ax1.plot(X, Z[1:-1], '--k')

ax2 = fig.add_subplot(2, 1, 2)  # divide the figure in 1 subplot
line2 = Line2D([], [], color='k')  # create a line of color blue, that will be used for the anomaed plot
ax2.add_line(line2)  # Add the corresponding line to the subplot
ax2.set_xlabel('x (in m)')  # Label the x axis
ax2.set_ylabel(r'q (in $m^2/s$)')  # Label the y axis
ax2.set_xlim([0, length])  # Fix the length of the x axis we will see
ax2.set_ylim([np.amin(q), 1.1 * np.amax(q)])  # Fix the length of the x axis we will see


# Define the function to initialize the animated plot
def init():
    line.set_data([], [])
    line2.set_data([], [])
    time_text1.set_text('')
    return line, line2, time_text1


# Define the function to plot each graph in the animated plot
def animate(i):
    line.set_data(X, Z[1:-1] + h[i, 1:-1])  # Everytime the function is called (so for i between 0
    # and its final value, prescribed later), plot h[i,:] as a function of X
    line2.set_data(X, q[i, 1:-1])  # Everytime the function is called (so for i between
    # 0 and its final value, prescribed later), plot h[i,:] as a function of X
    time_text1.set_text('time = %.0f s' % (i * dt))
    return line, line2, time_text1


# Create the animation plot on figure named fig, where at each i
# it will call the function animate(i), with an initialization function init,
# where i in animate(i) will run from 0 to frames = Nt. Interval denote the velocity (1/frame per second)
# at which the animation will run, and repeat = True means that the animation plot will be repeated once it finishes.
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=Nt, blit=True, interval=5., repeat=True)

# Show the figure/animated plot
plt.show()
################################################
