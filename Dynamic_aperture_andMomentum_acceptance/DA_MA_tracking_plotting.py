# %%
import xtrack as xt

RING = xt.Line.from_json('lattice_v1o5.json')

# %%
import DA_studies as da

# %%
emit_norm_x = 13000e-6    #m rad [normalized emittances from the positron production]
emit_norm_y = 12000e-6    #m rad [normalized emittances from the positron production]

# %%
import xobjects as xo

RING.discard_tracker()
RING.build_tracker(_context=xo.ContextCpu())
RING.configure_radiation(model='mean')
RING.compensate_radiation_energy_loss()

twiss_RING = RING.twiss(eneloss_and_damping=True , radiation_integrals=True)


found='DA'                            #or MA  , no value for energy spread needed to be given for DA
(x_normalized, y_normalized, delta_init, nn_x_theta, nn_y_r, num_delta, num_particles) = da.inital_conditions_grid (found, max_r_y=50,min_theta_x=-50, max_theta_x=50,num_r_y_points=20,num_theta_x_points=20)
px_normalized = 0 
py_normalized = 0
zeta = twiss_RING.zeta[0]
delta = delta_init + twiss_RING.delta[0]



# Match particles to the machine optics and orbit
particles = RING.build_particles(
    x_norm=x_normalized, px_norm=0,
    y_norm=y_normalized, py_norm=0,
    nemitt_x= emit_norm_x, nemitt_y=emit_norm_y, # normalized emittances
    zeta=zeta, delta=delta)

twa=RING.twiss(method='6d', eneloss_and_damping=True, radiation_integrals=True, compute_chromatic_properties=True)
twa.rows[:10].cols['betx bety']
print(particles.x, particles.y)

# %%
import os
import numpy as np

numcores = os.cpu_count()
num_turns = 2000

# Rebuild tracker with CPU multithreading
RING.discard_tracker()


# apply radiation damping
RING.build_tracker(_context=xo.ContextCpu(omp_num_threads=numcores))
RING.configure_radiation(model='mean')
RING.compensate_radiation_energy_loss()

particles.reorganize()

#  start tracking
print(" Starting particle tracking...")
RING.track(particles, num_turns=num_turns, time=True, with_progress=10)

#  sort to group lost particles last
particles.sort(interleave_lost_particles=True)

print(f" Tracking complete in {RING.time_last_track:.2f} seconds")
print(f" Particles survived: {np.sum(particles.state > 0)} / {particles._capacity}")

# %%
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def DA_vs_turns(particles, num_r_steps, num_theta_steps, x_norm, y_norm, delta_initial, delta_plots=False, emit_norm_x=None, emit_norm_y=None, gamma_rel=None, beta_x=None, beta_y=None):

    if isinstance(particles, dict):
        max_turns = np.shape(particles['x'])[1]-1 # minus 1 for the initial condition
        part_at_turn = np.nanmax(particles['at_turn'],axis=1)
    else:
        max_turns = np.max(particles.filter(particles.at_element==0).at_turn) # normally I should pass the maximum number (n_turn) of asked turns
        part_at_turn = particles.at_turn

    if delta_plots and np.size(delta_initial) > 1:

        for ii in np.unique(delta_initial):
            delta_index = np.where(delta_initial==ii)[0]
            
            x_norm_1d = x_norm[delta_index]
            y_norm_1d = y_norm[delta_index]
            part_at_turn_1d = part_at_turn[delta_index]
            x_norm_2d = x_norm_1d.reshape(num_r_steps, num_theta_steps)
            y_norm_2d = y_norm_1d.reshape(num_r_steps, num_theta_steps)
            part_at_turn_2d = part_at_turn_1d.reshape(num_r_steps, num_theta_steps)
                        
            x_DA = np.full(num_theta_steps, np.nan)
            y_DA = np.full(num_theta_steps, np.nan)
            for jj in range(num_theta_steps):
                for ii in range(num_r_steps):
                    if part_at_turn_2d[ii,jj] != max_turns:
                        x_DA[jj] = x_norm_2d[ii,jj]
                        y_DA[jj] = y_norm_2d[ii,jj]
                        break

            min_DA = np.nanmin(np.round(np.sqrt(x_DA**2+y_DA**2),1)) 
            where_min_DA = np.where(np.round(np.sqrt(x_DA**2+y_DA**2),1) == min_DA)[0]
            
            # Plot DA using scatter and pcolormesh
            fig = plt.subplots()
            plt.scatter(x_norm_1d, y_norm_1d, c=part_at_turn_1d)
            plt.plot(x_DA, y_DA, '-', color='r', label='DA for $\delta$=%.1E'%(ii))
            plt.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r', label='DA$_{min}$=%.1f$\sigma$'%(min_DA))
            plt.xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
            plt.ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_y}$]')
            cb = plt.colorbar()
            cb.set_label('Lost at turn')
            plt.legend(fontsize='small', loc='best')

            fig = plt.subplots()
            plt.pcolormesh(x_norm_2d, y_norm_2d, part_at_turn_2d, shading='gouraud')
            plt.plot(x_DA, y_DA, '-', color='r', label='DA for $\delta$=%.1E'%(ii))
            plt.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r', label='DA$_{min}$=%.1f$\sigma$'%(min_DA))    
            plt.xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
            plt.ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_y}$]')
            ax = plt.colorbar()
            ax.set_label('Lost at turn')
            plt.legend(fontsize='small', loc='best')
    
    else:

        if not delta_plots and np.size(delta_initial) > 1:
            closest_to_zero_delta = delta_initial[(np.abs(delta_initial - 0)).argmin()]
            delta_index = np.where(delta_initial==closest_to_zero_delta)[0]
            x_norm_1d = x_norm[delta_index]
            y_norm_1d = y_norm[delta_index]
            part_at_turn_1d = part_at_turn[delta_index]
        else:
            x_norm_1d = x_norm
            y_norm_1d = y_norm      
            part_at_turn_1d = part_at_turn

        x_norm_2d = x_norm_1d.reshape(num_r_steps, num_theta_steps)
        y_norm_2d = y_norm_1d.reshape(num_r_steps, num_theta_steps)
        part_at_turn_2d = part_at_turn_1d.reshape(num_r_steps, num_theta_steps)
        x_DA = np.full(num_theta_steps, np.nan)
        y_DA = np.full(num_theta_steps, np.nan)
        for jj in range(num_theta_steps):
            for ii in range(num_r_steps):
                if part_at_turn_2d[ii,jj] != max_turns:
                    x_DA[jj] = x_norm_2d[ii,jj]
                    y_DA[jj] = y_norm_2d[ii,jj]
                    break

        min_DA = np.nanmin(np.round(np.sqrt(x_DA**2+y_DA**2),1)) 
        where_min_DA = np.where(np.round(np.sqrt(x_DA**2+y_DA**2),1) == min_DA)[0]

        x_in_meters = x_norm_1d * np.sqrt(beta_x * emit_norm_x/gamma_rel)  # Convert normalized x to meters
        y_in_meters = y_norm_1d * np.sqrt(beta_y * emit_norm_y/gamma_rel)  # Convert normalized y to meters
        x_DA_meters = x_DA * np.sqrt(beta_x * emit_norm_x/gamma_rel)  # Convert normalized DA x to meters
        y_DA_meters = y_DA * np.sqrt(beta_y * emit_norm_y/gamma_rel)
        min_DA_meters = np.sqrt(x_DA_meters[where_min_DA[0]]**2 + y_DA_meters[where_min_DA[0]]**2)

        sqrt_2Jx_full = x_in_meters / np.sqrt(beta_x)
        sqrt_2Jy_full = y_in_meters / np.sqrt(beta_y)
        sqrt_2Jx = x_DA_meters / np.sqrt(beta_x)
        sqrt_2Jy = y_DA_meters / np.sqrt(beta_y)

        min_DA_meters_action_unit = np.sqrt(sqrt_2Jx_full[where_min_DA[0]]**2 + sqrt_2Jy_full[where_min_DA[0]]**2)

        fig, ax = plt.subplots()
        sc = ax.scatter(x_norm_1d, y_norm_1d, c=part_at_turn_1d)
        ax.plot(x_DA, y_DA, '-', color='r', label='DA')
        ax.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r',
        label=f'DA$_{{min}}$={min_DA:.2f}$\\sigma$')
        ax.set_xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
        ax.set_ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_y}$]')
        cb = fig.colorbar(sc, ax=ax, label='Lost at turn')
        ax.legend(fontsize='small', loc='best')
        

        fig, ax = plt.subplots()
        im = ax.pcolormesh(x_norm_2d, y_norm_2d, part_at_turn_2d,
                   shading='gouraud', linewidth=0, edgecolors='none')
        ax.plot(x_DA, y_DA, '-', color='r', label='DA')
        ax.plot(x_DA[where_min_DA], y_DA[where_min_DA], 'o', color='r',
        label=f'DA$_{{min}}$={min_DA:.2f}$\\sigma$')
        ax.set_xlabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$]')
        ax.set_ylabel(r'$\hat{y}$ [$\sqrt{\varepsilon_y}$]')
        cb = fig.colorbar(im, ax=ax, label='Lost at turn')
        ax.legend(fontsize='small', loc='best')


        fig, ax = plt.subplots()
        sc = ax.scatter(x_in_meters, y_in_meters, c=part_at_turn_1d)
        ax.plot(x_DA_meters, y_DA_meters, '-', color='r', label='DA in meters')
        ax.plot(x_DA_meters[where_min_DA], y_DA_meters[where_min_DA], 'o', color='r',
        label=f'DA$_{{min}}$ = {min_DA_meters:.3f}$m$')
        ax.set_xlabel(r'$A_x$ [m]')
        ax.set_ylabel(r'$A_y$ [m]')
        cb = fig.colorbar(sc, ax=ax, label='Lost at turn')
        ax.legend(fontsize='small', loc='best')
        

        fig, ax = plt.subplots()
        im = ax.pcolormesh(
        x_in_meters.reshape(num_r_steps, num_theta_steps),
        y_in_meters.reshape(num_r_steps, num_theta_steps),
        part_at_turn_1d.reshape(num_r_steps, num_theta_steps),
        shading='gouraud', linewidth=0, edgecolors='none')
        ax.plot(x_DA_meters, y_DA_meters, '-', color='r', label='DA in meters')
        ax.plot(x_DA_meters[where_min_DA], y_DA_meters[where_min_DA], 'o', color='r',
        label=f'DA$_{{min}}$ = {min_DA_meters:.3f}$m$')
        ax.set_xlabel(r'$A_x$ [m]')
        ax.set_ylabel(r'$A_y$ [m]')
        cb = fig.colorbar(im, ax=ax, label='Lost at turn')
        ax.legend(fontsize='small', loc='best')
        

        r_vals = np.linspace(0, 1, num_r_steps)
        theta_vals = np.linspace(0, 2*np.pi, num_theta_steps)

        R, THETA = np.meshgrid(r_vals, theta_vals, indexing='ij')  # shape = (num_r_steps, num_theta_steps)

        X_norm = R * np.cos(THETA)  # normalized Ax
        Y_norm = R * np.sin(THETA)  # normalized Ay

        sqrt_2Jx_grid = sqrt_2Jx_full.reshape(num_r_steps, num_theta_steps)
        sqrt_2Jy_grid = sqrt_2Jy_full.reshape(num_r_steps, num_theta_steps)
        Z = part_at_turn_1d.reshape(num_r_steps, num_theta_steps)

        fig, ax = plt.subplots()
        im = ax.pcolormesh(sqrt_2Jx_grid, sqrt_2Jy_grid, Z,
                   shading='gouraud', linewidth=0, edgecolors='none')
        ax.plot(sqrt_2Jx, sqrt_2Jy, '-', color='r', label='DA in action units')
        ax.plot(sqrt_2Jx[where_min_DA], sqrt_2Jy[where_min_DA], 'o', color='r',
        label=f'DA$_{{min}}$ = {min_DA_meters_action_unit:.6f}$\\sqrt{{m}}$')
        ax.set_xlabel(r'$\sqrt{2J_x}$ [$\sqrt{\mathrm{m}\cdot\mathrm{rad}}$]')
        ax.set_ylabel(r'$\sqrt{2J_y}$ [$\sqrt{\mathrm{m}\cdot\mathrm{rad}}$]')
        cb = fig.colorbar(im, ax=ax, label='Lost at turn')
        ax.legend(fontsize='small', loc='best')

    


    return (x_DA, y_DA, where_min_DA,x_DA_meters, y_DA_meters, min_DA_meters,sqrt_2Jx, sqrt_2Jy,sqrt_2Jx_full, sqrt_2Jy_full,min_DA_meters_action_unit)

def MA_vs_turns(particles, num_r_steps, num_delta_steps, x_norm, y_norm, delta_initial):

    if isinstance(particles, dict):
        max_turns = np.shape(particles['x'])[1]-1 # minus 1 for the initial condition
        part_at_turn = np.nanmax(particles['at_turn'],axis=1)
    else:
        max_turns = np.max(particles.filter(particles.at_element==0).at_turn) # normally I should pass the maximum number (n_turn) of asked turns
        part_at_turn = particles.at_turn

    theta = np.max(np.unique(np.arctan2(y_norm,x_norm)))

    x_norm_2d = x_norm.reshape(num_delta_steps, num_r_steps)
    y_norm_2d = y_norm.reshape(num_delta_steps, num_r_steps)
    delta_norm_2d = delta_initial.reshape(num_delta_steps, num_r_steps)
    part_at_turn = part_at_turn
    part_at_turn_2d = part_at_turn.reshape(num_delta_steps, num_r_steps)
    x_MA = np.full(num_delta_steps, np.nan)
    y_MA = np.full(num_delta_steps, np.nan)
    delta_MA = np.full(num_delta_steps, np.nan)
    for jj in range(num_delta_steps):
        for ii in range(num_r_steps):
            if part_at_turn_2d[jj,ii] != max_turns:
                x_MA[jj] = x_norm_2d[jj,ii]
                y_MA[jj] = y_norm_2d[jj,ii]
                delta_MA[jj] = delta_norm_2d[jj,ii]
                break

    min_MA = np.nanmin(np.round(np.sqrt(x_MA**2+y_MA**2),1)) 
    where_min_MA = np.where(np.round(np.sqrt(x_MA**2+y_MA**2),1) == min_MA)[0]


    fig, ax = plt.subplots()
    sc = ax.scatter(delta_initial*100, x_norm, c=part_at_turn)
    ax.plot(delta_MA*100, x_MA, '-', color='r', label='MA')
    ax.plot(delta_MA[where_min_MA]*100, x_MA[where_min_MA], 'o', color='r',
        label=f'MA$_{{min}}$={min_MA:.1f}$\\sigma$')
    ax.set_xlabel(r'$\delta$ [%]')
    ax.set_ylabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$], $\hat{y}$ [Tan(%.1f)$\sqrt{\varepsilon_y}$]' % (theta*180/np.pi))
    cb = fig.colorbar(sc, ax=ax, label='Lost at turn')
    ax.legend(fontsize='small', loc='best')
    

    fig, ax = plt.subplots()
    im = ax.pcolormesh(delta_norm_2d*100, x_norm_2d, part_at_turn_2d,
                   shading='gouraud', linewidth=0, edgecolors='none')
    ax.plot(delta_MA*100, x_MA, '-', color='r', label='MA')
    ax.plot(delta_MA[where_min_MA]*100, x_MA[where_min_MA], 'o', color='r',
        label=f'MA$_{{min}}$={min_MA:.1f}$\\sigma$')
    ax.set_xlabel(r'$\delta$ [%]')
    ax.set_ylabel(r'$\hat{x}$ [$\sqrt{\varepsilon_x}$], $\hat{y}$ [Tan(%.1f)$\sqrt{\varepsilon_y}$]' % (theta*180/np.pi))
    cb = fig.colorbar(im, ax=ax, label='Lost at turn')
    ax.legend(fontsize='small', loc='best')
    


    return (x_MA, delta_MA, where_min_MA)
    



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# %%
DA_vs_turns(particles, num_r_steps=20, num_theta_steps=20, x_norm=x_normalized, y_norm=y_normalized, delta_initial=delta_init, delta_plots=False,
 emit_norm_x=13000e-6,                 # normalized emittance in x [m·rad] linac 
    emit_norm_y=12000e-6,                 # normalized emittance in y [m·rad] linac
    gamma_rel=twiss_RING.gamma0,      # relativistic gamma
    beta_x=twiss_RING.betx[0],        # Twiss β_x at injeprint(f"beta functions at the start of the ring:")
    beta_y=twiss_RING.bety[0],      # Twiss β_y part_at_turn_2d = part_at_turn_1d.reshape(num_r_steps, num_theta_steps) at injection
)


# %%
print(f"hor emit : {twiss_RING.rad_int_eq_gemitt_x}") 


# %%
# print(f"Starting point s-position: {twiss_RING.s[0]}")
# print(f"Beta functions at s=0: βx={twiss_RING.betx[0]:.3f}, βy={twiss_RING.bety[0]:.3f}")
# print(f"Equilibrium emittance: {twiss_RING.rad_int_eq_gemitt_x*1e9:.3f} nm⋅rad")

# %%
RING.discard_tracker()
RING.build_tracker(_context=xo.ContextCpu())
RING.configure_radiation(model='mean')
RING.compensate_radiation_energy_loss()


energy_spread = 7.245514e-04          # ~ 0.07% energy spread

found='MA' #or MA  , no value for energy spread in DA
(x_normalized, y_normalized, delta_init, nn_x_theta, nn_y_r, num_delta, num_particles) = da.inital_conditions_grid (found, max_r_y=50,min_theta_x=np.pi/4, max_theta_x=np.pi/4,num_r_y_points=31,num_theta_x_points=1, energy_spread=energy_spread,delta_initial_values=np.linspace(-45*energy_spread, 45*energy_spread, 51))
px_normalized = 0 
py_normalized = 0
zeta = twiss_RING.zeta[0]
delta = delta_init + twiss_RING.delta[0]



# Match particles to the machine optics and orbit
particles = RING.build_particles(
    x_norm=x_normalized, px_norm=0,
    y_norm=y_normalized, py_norm=0,
    nemitt_x= emit_norm_x, nemitt_y=emit_norm_y, # normalized emittances
    zeta=zeta, delta=delta)


# %%
particles.reorganize()

# # Optional: activate multi-core CPU parallelization

RING.discard_tracker()
RING.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))
RING.configure_radiation(model='mean')
RING.compensate_radiation_energy_loss()

# Track
RING.track(particles, num_turns=2000, time=True, with_progress=10)

print(f'Tracked in {RING.time_last_track} seconds')


particles.sort(interleave_lost_particles=True)



# %%
MA_vs_turns(particles, nn_y_r, num_delta, x_normalized, y_normalized, delta_init)  

# %%



