import matplotlib.pyplot as plt
import numpy as np

#K is the viscosity parameter
K = .25

def main():
    fine = int(input("Cells per x value (positive integer): "))
    time = int(input("Number of iterations: "))

    #these are the average values for each cell
    #change initial conditions here
    left_rho = [(1+np.sin(float(x)/fine+0.5/fine))/2 for x in range(100*fine)]
    #left_rho = [(0.3+0.002*x/fine) for x in range(100*fine)]
    left_v = [1-x for x in left_rho]
    pgd_rho = [0.3 for _ in range(20*fine)]
    pgd_v = [0.2 for _ in range(20*fine)]
    right_rho = [0.3 for _ in range(100*fine)]
    right_v = [0.7 for _ in range(100*fine)]

    full_rho =  left_rho + pgd_rho + right_rho
    full_v = left_v+ pgd_v + right_v
    
    x_values = [x/fine for x in range(220*fine)]
    plt.plot(x_values,full_rho,'#3399FF')

    for t in range(time):

        left_bounds = (left_rho[0],left_v[0],left_rho[-1],left_v[-1])
        pgd_bounds  = (pgd_rho[0],pgd_v[0],pgd_rho[-1],pgd_v[-1])
        right_bounds = (right_rho[0],right_v[0],right_rho[-1],right_v[-1])

        left_rho.append(pgd_bounds[0])
        left_v.append(pgd_bounds[1])
        left_rho.insert(0,right_bounds[2])
        left_v.insert(0,right_bounds[3])

        pgd_rho.append(right_bounds[0])
        pgd_v.append(right_bounds[1])
        pgd_rho.insert(0,left_bounds[2])
        pgd_v.insert(0,left_bounds[3])

        right_rho.append(left_bounds[0])
        right_v.append(left_bounds[1])
        right_rho.insert(0,pgd_bounds[2])
        right_v.insert(0,pgd_bounds[3])

        ar_left = ar_step(left_rho,left_v,.5)
        left_rho = ar_left[0]
        left_v = ar_left[1]

        pgd = pgd_step(pgd_rho,pgd_v,.5)
        pgd_rho = pgd[0]
        pgd_v = pgd[1]

        ar_right = ar_step(right_rho,right_v,.5)
        right_rho = ar_right[0]
        right_v = ar_right[1]


    full_rho = left_rho + pgd_rho + right_rho
    full_v = left_v+ pgd_v + right_v
    plt.plot(x_values,full_rho,'#FF0000')
    plt.show()
    

#McCormack scheme with artificial viscosity smoothing
#length of arrays should be equal and at least 3
def ar_step (rho, v, dtdx):
    size = len(rho)
    U = []
    for x in range(size):
        U.append(np.array([rho[x],(rho[x]*(rho[x]+v[x]))]))

    F = [v[x]*U[x] for x in range(size)]
    
    visc = []
    visc.append(np.linalg.norm(U[1]-U[0])/(np.linalg.norm(U[1])+np.linalg.norm(U[0])))
    for x in range(1,(size-1)):
        visc.append(np.linalg.norm(U[x+1]-2*U[x]+U[x-1])/(np.linalg.norm(U[x+1])+2*np.linalg.norm(U[x])+np.linalg.norm(U[x-1])))
    visc.append(np.linalg.norm(U[-1]-U[-2])/(np.linalg.norm(U[-1])+np.linalg.norm(U[-2])))

    #predictor step
    U_pred = [U[x]-dtdx*(F[x]-F[x-1]) for x in range(1,size)]
    v_pred = [(u[1]/u[0] - u[0]) if u[0] !=0 else 0 for u in U_pred]
    F_pred = [v_pred[x]*U_pred[x] for x in range(size-1)]

    #corrector step
    U_corr = [(0.5*(U[x]+U_pred[x])-0.5*dtdx*(F_pred[x+1]-F_pred[x])) for x in range(size-2)]
    #add prior values at boundaries for use in next step
    U_corr.insert(0,U[0])
    U_corr.append(U[-1])

    #viscosity step
    U_av = []
    for x in range(1,size-1):
        viscL = K*max(visc[x-1],visc[x])
        viscR = K*max(visc[x],visc[x+1])
        U_av.append(U_corr[x]+viscR*(U_corr[x+1]-U_corr[x])-viscL*(U_corr[x]-U_corr[x-1]))

    #returns a tuple of rho and v lists
    rho_new = [min(max(u[0],0.0001),1.) for u in U_av]
    v_new = [min(max((u[1]/u[0] - u[0]),0.),1.) for u in U_av]
    rho_v_tup = (rho_new,v_new)
    return rho_v_tup
    

#Godunov-type scheme
def pgd_step (rho, v, dtdx):
    size = len(rho)
    Q = []
    for x in range(size):
        Q.append(np.array([rho[x],(rho[x]*v[x])]))


    F = [v[x]*Q[x] for x in range(size)]

    #trying it without correction term
    Q_next = [(Q[x] - dtdx*(F[x]-F[x-1])) for x in range(1,size-1)]

    #returns a tuple of rho and v lists
    #bounds rho above to avoid overflow and make the graph readable
    rho_new = [min(max(q[0],0.0001),100) for q in Q_next]
    v_new = [max(q[1]/q[0],0.) if q[0] !=0 else 1 for q in Q_next]
    rho_v_tup = (rho_new,v_new)
    return rho_v_tup




if __name__ == "__main__":
    main()