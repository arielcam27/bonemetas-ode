# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import ode
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rc('font', family='sans-serif')
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]  


def RH(a3,b3,c1,c2,c3,c4):
    B1 = b1/a3
    B2 = b2/a3
    B3 = b3/a3
    
    K1 = c1*k/a3
    K2 = c2*k/a3
    K3 = c3/a2
    K4 = c4/a1
    
    # eq libre
    v3 = 0.
    v1 = (B2)**(1./g2)
    v2 = (B1)**(1./g1)
    
    u1eq = v1*(a3/a2)**(1./g2)
    u2eq = v2*(a3/a1)**(1./g1)
    u3eq = v3*k
    
    # eq invasion
    vv3 = (-B3 + 1. + K3*B2 + K4*B1)/(1. + K2*K3 + K1*K4)
    vv1 = (B2 - K2*vv3)**(1./g2)
    vv2 = (B1 - K1*vv3)**(1./g1)
    
    uu1eq = vv1*(a3/a2)**(1./g2)
    uu2eq = vv2*(a3/a1)**(1./g1)
    uu3eq = vv3*k
    
    # a's de RH (a son las alphas, no confundirse)
    aa1 = -B1*K4 \
         -B2*K3 \
         +B3 \
         +K1*K4*v3 \
         +K2*K3*v3 \
         +2.*v3 \
         -1.
    aa2 = -B1*B2*g1*g2 \
         -B1*K2*K4*g1*v3 \
         +B1*K2*g1*g2*v3 \
         -B2*K1*K3*g2*v3 \
         +B2*K1*g1*g2*v3 \
         +K1*K2*K3*g2*v3**2 \
         +K1*K2*K4*g1*v3**2 \
         -K1*K2*g1*g2*v3**2
    aa3 = +B1**2*B2*K4*g1*g2 \
         -B1**2*K2*K4*g1*g2*v3 \
         +B1*B2**2*K3*g1*g2 \
         -B1*B2*B3*g1*g2 \
         -3.*B1*B2*K1*K4*g1*g2*v3 \
         -3.*B1*B2*K2*K3*g1*g2*v3 \
         -2.*B1*B2*g1*g2*v3 \
         +B1*B2*g1*g2 \
         +B1*B3*K2*g1*g2*v3 \
         +3.*B1*K1*K2*K4*g1*g2*v3**2 \
         +2.*B1*K2**2*K3*g1*g2*v3**2 \
         +2.*B1*K2*g1*g2*v3**2 \
         -B1*K2*g1*g2*v3 \
         -B2**2*K1*K3*g1*g2*v3 \
         +B2*B3*K1*g1*g2*v3 \
         +2.*B2*K1**2*K4*g1*g2*v3**2 \
         +3.*B2*K1*K2*K3*g1*g2*v3**2 \
         +2.*B2*K1*g1*g2*v3**2 \
         -B2*K1*g1*g2*v3 \
         -B3*K1*K2*g1*g2*v3**2 \
         -2.*K1**2*K2*K4*g1*g2*v3**3 \
         -2.*K1*K2**2*K3*g1*g2*v3**3 \
         -2.*K1*K2*g1*g2*v3**3 \
         +K1*K2*g1*g2*v3**2
    
    RH1 = aa1*aa2 - aa3
    RH2 = aa1
    RH3 = aa3
    
    print('equilibrio libre (OC-OB-CC):')
    print(u1eq)
    print(u2eq)
    print(u3eq)
    
    print('equilibrio inv (OC-OB-CC):')
    print(uu1eq)
    print(uu2eq)
    print(uu3eq)
    
    print('criterio RH (equilibrio libre):')
    print(RH1)
    print(RH2)
    print(RH3)
    
    lefthand = b2*c3/a2 + b1*c4/a1
    righthand = b3 - a3
    print('Teorema 2:')
    print(lefthand<righthand)
    return u1eq, u2eq, u3eq


def my_odeint(f, y0, t):    
    y0 = np.asarray(y0)
    
    solver = ode(f)
    solver.set_integrator('dop853', max_step=0.01)
    
    t0 = t[0]
    t_final = t[-1]
    
    solver.set_initial_value(y0, t0)
    
    y_result = [y0]
    
    i = 1
    current_t = t[i]
    
    while solver.successful() and solver.t < t_final:
        solver.integrate(current_t, step=1)
        i += 1
        if i < len(t):
            current_t = t[i]
    
        y_result.append(solver.y)
    
    return np.array(y_result)

def simple_plots(tt,uu0,uu1,uu2,uu3):
    ax1 = plt.subplot(221)
    plt.grid()
    ax1.plot(tt, uu0)
    
    ax2 = plt.subplot(222)
    plt.grid()
    ax2.plot(tt, uu1)
    
    ax3 = plt.subplot(223)
    plt.grid()
    ax3.plot(tt, uu2)
    
    ax4 = plt.subplot(224)
    plt.grid()
    ax4.plot(tt, uu3)

def fancy_plots(tt,uu0,uu1,uu2,uu3,plotname):
    fig, (ax0,ax1,ax2,ax3) = plt.subplots(nrows=4,figsize=(5, 5), dpi=1000,sharex=True)

    ax0.plot(tt, uu0, 'b', linewidth=1)
    ax0.set_ylabel('OC')           
#    ax0.set_ylim([-2.0,20]) 
    ax0.yaxis.grid(which="major")
    
    ax1.plot(tt, uu1, 'g', linewidth=1)
    ax1.set_ylabel('OB')
#    ax1.set_ylim([-200.,5000])
    ax1.yaxis.grid(which="major")
    
    ax2.plot(tt, uu2, 'r', linewidth=1)
    ax2.set_ylabel('CC')
#    ax2.set_ylim([-5,60])
    ax2.yaxis.grid(which="major")
    
    ax3.plot(tt, uu3, 'k', linewidth=1)
    ax3.set_ylabel('BM')
    ax3.set_xlabel('Time t')
#    ax3.set_ylim([85,100])
    ax3.set_xlim([0,2000])
    ax3.yaxis.grid(which="major")
    
    plt.savefig(plotname+'.pdf', bbox_inches='tight')
    
# fixed parameters
a1 = 0.3
a2 = 0.1
b1 = 0.2
b2 = 0.02
g1 = -0.3
g2 = 0.5
k = 300.0
t_output = np.arange(0, 2000, 0.01)

def periodicity1():
    a3 = 0.045
    b3 = 0.05
    c1 = 0.001
    c2 = -0.00005
    c3 = 0.005
    c4 = 0.0
    k1 = 0.08
    k2 = 0.0015
    u0 = [10.0, 5.0, 50.0, 92.0]
    def ode_eqs(t, x_vec):
        u1, u2, u3, z = x_vec
        du1 = a1*u1*u2**g1 - b1*u1 + c1*u1*u3
        du2 = a2*u1**g2*u2 - b2*u2 + c2*u2*u3
        du3 = a3*u3*(1.0 - u3/k) - b3*u3 + c3*u1**g2*u3 + c4*u2**g1*u3   
        dz = -k1*(max(u1 - u1eq,0.0))**0.5 + k2*(max(u2 - u2eq,0.0))**0.5
        return [du1, du2, du3, dz]
    u1eq, u2eq, u3eq = RH(a3,b3,c1,c2,c3,c4)
    y_result = my_odeint(ode_eqs, u0, t_output)
    u = y_result.T
    
    simple_plots(t_output, u[0], u[1], u[2], u[3])
    fancy_plots(t_output, u[0], u[1], u[2], u[3], 'periodicity1')
    
def periodicity2():
    a3 = 0.055
    b3 = 0.05
    c1 = 0.001
    c2 = -0.00005
    c3 = 0.005
    c4 = -0.015
    k1 = 0.045
    k2 = 0.0015
    u0 = [10.0, 5.0, 20.0, 95.0]
    def ode_eqs(t, x_vec):
        u1, u2, u3, z = x_vec
        du1 = a1*u1*u2**g1 - b1*u1 + c1*u1*u3
        du2 = a2*u1**g2*u2 - b2*u2 + c2*u2*u3
        du3 = a3*u3*(1.0 - u3/k) - b3*u3 + c3*u1**g2*u3 + c4*u2**g1*u3   
        dz = -k1*(max(u1 - u1eq,0.0))**0.5 + k2*(max(u2 - u2eq,0.0))**0.5
        return [du1, du2, du3, dz]
    u1eq, u2eq, u3eq = RH(a3,b3,c1,c2,c3,c4)
    y_result = my_odeint(ode_eqs, u0, t_output)
    u = y_result.T
    
    simple_plots(t_output, u[0], u[1], u[2], u[3])
#    fancy_plots(t_output, u[0], u[1], u[2], u[3], 'periodicity2')
    
def mixed():
    a3 = 0.055
    b3 = 0.05
    c1 = 0.001
    c2 = -0.005
    c3 = 0.001
    c4 = 0.0
    k1 = 0.023
    k2 = 0.0023
    u0 = [10.0, 5.0, 1.0, 90.0]
    def ode_eqs(t, x_vec):
        u1, u2, u3, z = x_vec
        du1 = a1*u1*u2**g1 - b1*u1 + c1*u1*u3
        du2 = a2*u1**g2*u2 - b2*u2 + c2*u2*u3
        du3 = a3*u3*(1.0 - u3/k) - b3*u3 + c3*u1**g2*u3 + c4*u2**g1*u3   
        dz = -k1*(max(u1 - u1eq,0.0))**0.5 + k2*(max(u2 - u2eq,0.0))**0.5
        return [du1, du2, du3, dz]
    u1eq, u2eq, u3eq = RH(a3,b3,c1,c2,c3,c4)
    y_result = my_odeint(ode_eqs, u0, t_output)
    u = y_result.T
    
    simple_plots(t_output, u[0], u[1], u[2], u[3])
    fancy_plots(t_output, u[0], u[1], u[2], u[3], 'mixed')
    
def osteolytic():
    a3 = 0.055
    b3 = 0.05
    c1 = 0.0005
    c2 = -0.009
    c3 = 0.001
    c4 = 0.0
    k1 = 0.02
    k2 = 0.003
    u0 = [10.0, 5.0, 1.0, 90.0]
    def ode_eqs(t, x_vec):
        u1, u2, u3, z = x_vec
        du1 = a1*u1*u2**g1 - b1*u1 + c1*u1*u3
        du2 = a2*u1**g2*u2 - b2*u2 + c2*u2*u3
        du3 = a3*u3*(1.0 - u3/k) - b3*u3 + c3*u1**g2*u3 + c4*u2**g1*u3   
        dz = -k1*(max(u1 - u1eq,0.0))**0.5 + k2*(max(u2 - u2eq,0.0))**0.5
        return [du1, du2, du3, dz]
    u1eq, u2eq, u3eq = RH(a3,b3,c1,c2,c3,c4)
    y_result = my_odeint(ode_eqs, u0, t_output)
    u = y_result.T
    
    simple_plots(t_output, u[0], u[1], u[2], u[3])
    fancy_plots(t_output, u[0], u[1], u[2], u[3], 'osteolytic')

periodicity1()
#periodicity2()
#mixed()
#osteolytic()