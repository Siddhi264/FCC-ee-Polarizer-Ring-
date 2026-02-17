# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import xobjects as xo
import xtrack as xt
import xpart as xp
import pandas as pd
import xfields as xf
import scipy.constants as sc
import matplotlib.colors as mcolors
%matplotlib widget
import os, re
import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython.display import display, HTML


# Ring layout:
# - 3 arcs × (2 suppressor cells + 14 arc cells + 2 suppressor cells) = 54 cells
# arc cell : 2 dipoles, 4 drifts
# suppressor cell 1 : 0 dipoles, 2 drifts  (missing bend cell)
# suppressor cell 2 : 2 dipoles, 4 drifts


#  Environment and particle setup
env = xt.Environment()
env.particle_ref = xt.Particles(kinetic_energy0=2.86e9, mass0 = xp.ELECTRON_MASS_EV)
env.vars.default_to_zero = True

env['l_quad'] = 0.45   #quadrupole length  (all quadrupoles are the same length for now)
env['l_dipole'] = 1.5  #bending length
env['ang_mb'] = 2* np.pi / 102  #bending angle (!! 102 bending magnets here intead of 96 as in design 1 ['half cell' with missing bend])
env['l_fodo'] = 5.00435  #length of each FODO cell 


#  Drift values based on 5.00435 m per arc cell
env['d_arc'] = (env['l_fodo'] - 2 * env['l_dipole'] - 2 * env['l_quad']) / 4


#  Quadrupole strengths (initial guesses pre matching)
env['kqf_arc'] = 0.95
env['kqd_arc'] = -0.95


# element definitions
env.new('dr_arc', xt.Drift, length='d_arc')
env.new('mq', xt.Quadrupole, length='l_quad')

env.new('qf_arc', 'mq', k1='kqf_arc')
env.new('qd_arc', 'mq', k1='kqd_arc')

env.new('half_qf_arc', xt.Quadrupole, length=env['l_quad'] / 2, k1='kqf_arc')
env.new('half_qd_arc', xt.Quadrupole, length=env['l_quad'] / 2, k1='kqd_arc')


env.new('mb', xt.RBend, length='l_dipole', angle='ang_mb', k0_from_h=True)


#  Arc cell ( 2 dipoles, 4 drifts)
# half QF_arc – drift – bend – drift – QD_arc - drift - bend - drift - half QF_arc)
arc_cell = env.new_line(components=[
    env.place('half_qf_arc'),
    env.place('dr_arc'), env.place('mb'),
    env.place('dr_arc'),
    env.place('qd_arc'), env.place('dr_arc'),env.place('mb'),
     env.place('dr_arc'), env.place('half_qf_arc')
])


def make_arc_cell(idx):
    return env.new_line(components=[

        *[arc_cell.replicate(f'arc_{idx}_cell_{i}') for i in range(1)],

    ])

arc_cell_comp = make_arc_cell(1)
#arc_cell_comp.survey().plot()

my_twiss = arc_cell_comp.twiss4d()
my_twiss.plot()

#matching the tunes
opt_tune = arc_cell_comp.match(
    solve=False, # <- prepare the match without running it
    compute_chromatic_properties=False,
    method='4d',
    vary=[
        xt.Vary('kqf_arc', limits=(0, 10),  step=1e-3),
        xt.Vary('kqd_arc', limits=(-10, 0), step=1e-3),
    ],
    targets=[
        xt.TargetSet(qx=1/4, qy=1/4, tol=1e-6),                         # pi/2 PHASE ADVANCE
    ]
)
opt_tune.target_status()

opt_tune.run_jacobian(30)

opt_tune.target_status()

opt_tune.vary_status()

#Optics inspection
tw = arc_cell_comp.twiss4d()
pl = tw.plot()
pl.ylim(left_hi=10, right_lo=-2, right_hi=2,
        lattice_hi=1.5, lattice_lo=-7)

tw.to_pandas()[['s','name','betx','bety','alfx','alfy','mux','muy','dx' , 'dy','dpx','dpy']]


# %%
#additional quadrupole strengths (suppressor and striaght regions)
env['kqd_sup_a'] = -0.962121
env['kqf_str_a'] = 0.989586
env['kqf_str_b'] = 1.01893
env['kqd_str_a'] = -1.03206
env['kqf_sup'] = 0.955138
env['kqd_sup_b'] = -0.9435615
env['KQF_str_doublet'] = 3.0
env['KQD_str_doublet'] = -3.2
env['kqf_sup_b'] = 0.95
env['KQF_str_triplet'] = 1.45
env['KQD_str_triplet'] = -0.844


# element definitions of above quads
env.new('qd_sup_a' , 'mq' , k1='kqd_sup_a')
env.new('half_qd_sup_a' , xt.Quadrupole , length=env['l_quad'] / 2 , k1='kqd_sup_a')

env.new('qf_str_a' , 'mq' , k1='kqf_str_a')
env.new('half_qf_str_a' , xt.Quadrupole , length=env['l_quad'] / 2 , k1='kqf_str_a')


env.new('qf_str_b' , 'mq' , k1='kqf_str_b')
env.new('half_qf_str_b' , xt.Quadrupole , length=env['l_quad'] / 2 , k1='kqf_str_b')

env.new('qd_str_a' , 'mq' , k1='kqd_str_a')
env.new('half_qd_str_a' , xt.Quadrupole , length=env['l_quad'] / 2 , k1='kqd_str_a')

env.new('qf_sup' , 'mq' , k1='kqf_sup')
env.new('half_qf_sup' , xt.Quadrupole , length=env['l_quad'] / 2 , k1='kqf_sup')

env.new('qd_sup_b' , 'mq' , k1='kqd_sup_b')
env.new('half_qd_sup_b' , xt.Quadrupole , length=env['l_quad'] / 2 , k1='kqd_sup_b')

env.new('QF_str_triplet', 'mq', k1='KQF_str_triplet', length=env['l_quad'])
env.new('half_QF_str_triplet', xt.Quadrupole, length=env['l_quad'] / 2, k1='KQF_str_triplet')


env.new('QD_str_triplet', 'mq', k1='KQD_str_triplet', length=env['l_quad'])
env.new('half_QD_str_triplet', xt.Quadrupole, length=env['l_quad'] / 2, k1='KQD_str_triplet')


env.new('qf_sup_b', 'mq', k1='kqf_sup_b', length=env['l_quad'])
env.new('half_qf_sup_b', xt.Quadrupole, length=env['l_quad'] / 2, k1='kqf_sup_b')

env.new('QF_str_doublet', 'mq', k1='KQF_str_doublet', length=env['l_quad'])
env.new('half_QF_str_doublet', xt.Quadrupole, length=env['l_quad'] / 2, k1='KQF_str_doublet')

env.new('QD_str_doublet', 'mq', k1='KQD_str_doublet', length=env['l_quad'])
env.new('half_QD_str_doublet', xt.Quadrupole, length=env['l_quad'] / 2, k1='KQD_str_doublet')



#sextupoles 

env['l_sext'] = 0.15  # [m] 
l_sext = env['l_sext']

env['k2_sf'] = 5.0
env['k2_sd'] = -5.0


env.new('sf', xt.Sextupole, k2='k2_sf', length=l_sext)
env.new('sd', xt.Sextupole, k2='k2_sd', length=l_sext)



# drift element definitions

#arc cell drift after adding sextupoles
env['d_arc_after_adding_sext'] = env['d_arc'] - env['l_sext'] 
env.new('dr_arc_after_adding_sext', xt.Drift, length='d_arc_after_adding_sext')
# also define half of dr_arc_after_adding_sext to place sextupoles symmetrically in the middle of the drift
env['d_arc_after_adding_sext_half'] = env['d_arc_after_adding_sext'] / 2
env.new('dr_arc_after_adding_sext_half', xt.Drift, length='d_arc_after_adding_sext_half')

#drift in the missing bend suppressor cell
env['d_sup1'] =(env['l_fodo'] - 2 * env['l_quad']) / 2
env.new('dr_sup1' , xt.Drift , length='d_sup1')

#drift in the suppressor cell 2
env['d_sup2'] =  (env['l_fodo'] - 2 * env['l_dipole'] - 1.5 * env['l_quad']) / 4  
env.new('dr_sup2' , xt.Drift , length='d_sup2')

env['d_sup3'] = 0.5 + (env['l_fodo'] - 2 * env['l_dipole'] - 1.5 * env['l_quad']) / 4  
env.new('dr_sup3' , xt.Drift , length='d_sup3')
#drift in the straight cell
env['d_str'] = (env['l_fodo'] - 2 * env['l_quad']) / 2
env.new('dr_str' , xt.Drift , length='d_str')

#to place the RF cavity in the centre of a drift, we also defne a half drift (half the length of the dr_str)
env['d_str_half'] = env['d_str'] / 2
env.new('dr_str_half' , xt.Drift , length ='d_str_half')

env['straight_section_length'] = 25.02175
straight_section_length = env['straight_section_length']

env['d_between_quads_triplet'] = 0.25 # [m] distance between given 2 quads where there is the  triplet quadrupoles in the straight section
env.new('dr_between_quads_triplet', xt.Drift, length='d_between_quads_triplet')

env['d_full_str_between_two_triplet_pairs'] = straight_section_length - 8 * env['d_between_quads_triplet'] - 13 * env['l_quad']
env.new('dr_full_str_between_two_triplet_pairs', xt.Drift, length='d_full_str_between_two_triplet_pairs')
#half of above
env['d_full_str_between_two_triplet_pairs_half'] = (env['d_full_str_between_two_triplet_pairs'] / 8 ) 
env.new('dr_full_str_between_two_triplet_pairs_half', xt.Drift, length='d_full_str_between_two_triplet_pairs_half')


# adding markers in the 2nd drift of the straight section (the one which does not host wigglers; and is proposed to host rf)
L = env['d_full_str_between_two_triplet_pairs_half']
n_slices = 100  # for 10% steps
slice_length = L / n_slices

new_drifts_with_markers = []

for i in range(n_slices):
    # Define drift segment
    drift_name = f"dr_str_triplet_slice_{i+1}"
    env.new(drift_name, xt.Drift, length=slice_length)
    new_drifts_with_markers.append(env.place(drift_name))
    
    # Add marker *after* drift, but only if not the last drift
    if i < n_slices - 1:
        marker_name = f"marker_str_triplet_{(i+1)*10}pct"
        env.new(marker_name, xt.Marker)
        new_drifts_with_markers.append(env.place(marker_name))


env['d_after_adding_wig'] = (((env['d_full_str_between_two_triplet_pairs'] / 4) - 3.8 ) / 2)   # Adjusted drift length after adding wigglers
env.new('dr_after_adding_wig', xt.Drift, length='d_after_adding_wig')

L_slicing_after_adding_wig = env['d_after_adding_wig']
n_slices_after_adding_wig = 100  # for 10% steps
slice_length_after_adding_wig = L_slicing_after_adding_wig / n_slices_after_adding_wig


# adding markers in the 1st drift of the straight section which hosts the wigglers
new_drifts_with_markers_wig = []

for j in range(n_slices_after_adding_wig):
    # Define drift segment
    drift_name_after_adding_wig = f"dr_str_triplet_slice_wig_{j+1}"
    env.new(drift_name_after_adding_wig, xt.Drift, length=slice_length_after_adding_wig)
    new_drifts_with_markers_wig.append(env.place(drift_name_after_adding_wig))
    
    # Add marker *after* drift, but only if not the last drift
    if j < n_slices_after_adding_wig - 1:
        marker_name_after_adding_wig = f"marker_str_triplet_wig_{(j+1)*10}pct"
        env.new(marker_name_after_adding_wig, xt.Marker)
        new_drifts_with_markers_wig.append(env.place(marker_name))



#wigglers
n_periods = 19
length_short = 0.05  # 5 cm for short segment
length_long = 0.15  # 15 cm for long segment
total_wig_length = length_short/2 + n_periods*length_long + (n_periods-1)*length_short + length_short/2

#wiggler function
def build_wiggler(name_prefix, n_periods, length_short, B_short, ratio, env):
    """
    Build an asymmetric wiggler with half-short segments at each end.

    Args:
        name_prefix: Name prefix for wiggler elements
        n_periods: Controls number of long segments (n_periods long segments)
        length_short: Length of short segment
        B_short: Magnetic field of short segment
        ratio: Ratio of long/short segment lengths
        env: xtrack environment
    """

    length_long = ratio * length_short
    B_long = -B_short * (length_short / length_long)  # ensures integral B.dl = 0

    p0c_GeV = float(env.particle_ref.p0c) * 1e-9  # [GeV]
    rho_short = p0c_GeV / (0.2998 * B_short)
    rho_long = p0c_GeV / (0.2998 * abs(B_long))

    theta_short = length_short / rho_short
    theta_long = length_long / rho_long

    # Calculate number of complete magnetic periods (L-S pairs)
    

    # Print key parameters for sanity check
    print(f"--- Asymmetric Wiggler Parameters: {name_prefix} ---")
    print(f"Short segment: length={length_short:.6f}m, B={B_short:.6f}T, θ={theta_short:.6f}rad")
    print(f"Long segment:  length={length_long:.6f}m, B={B_long:.6f}T, θ={theta_long:.6f}rad")
    print(f"Full period: {length_short + length_long:.6f}m")


    # Calculate net angle and field integral
    net_angle_per_period = theta_short - theta_long
    net_field_integral = B_short * length_short + B_long * length_long

    print(f"Net angle per period: {net_angle_per_period:.9f}rad")
    print(f"Net field integral per period: {net_field_integral:.9f}T·m")

    # Calculate total structure parameters
    total_angle = theta_short/2 + n_periods*(-theta_long) + (n_periods-1)*theta_short + theta_short/2

    print(f"Total structure length: {total_wig_length:.6f}m")
    print(f"Total bend angle: {total_angle:.9f}rad")
    print(f"----------------------------------------")

    segments = []

    # First half-period (half of short segment)
    name = f"{name_prefix}half_of_short_start"
    

    env.new(name, xt.RBend, length=length_short/2, h=0, k0=theta_short/length_short, k0_from_h=False)
    segments.append(env.place(name))

    # Full periods
    for i in range(n_periods):
        # Long segment
        name = f"{name_prefix}_long_{i}"
        env.new(name, xt.RBend, length=length_long, h=0, k0=-theta_long/length_long, k0_from_h=False)
        segments.append(env.place(name))

        # Short segment (except for the last period)
        if i < n_periods - 1:
            name = f"{name_prefix}_short_{i+1}"
            env.new(name, xt.RBend, length=length_short, h=0, k0=theta_short/length_short, k0_from_h=False)
            segments.append(env.place(name))

    # Last half-period (half of short segment)
    name = f"{name_prefix}half_of_short_end"

    env.new(name, xt.RBend, length=length_short/2, h=0 , k0=theta_short/length_short, k0_from_h=False)
    segments.append(env.place(name))

    return env.new_line(components=segments)


# Creating the wiggler sections with proper period count
wiggler_section_1 = build_wiggler(
    name_prefix="wig1",
    n_periods=19,                # Creates 7 complete periods 
    length_short=0.05,          # 5 cm for short segment
    B_short=1.5,                # 2 Tesla peak field
    ratio=3.0,                  # long = 3 × short
    env=env
)

wiggler_section_2 = build_wiggler(
    name_prefix="wig2",
    n_periods=19,                # Creates 7 complete periods 
    length_short=0.05,          # 5 cm for short segment
    B_short=1.5,                # 2 Tesla peak field
    ratio=3.0,                  # long = 3 × short
    env=env
)

wiggler_section_3 = build_wiggler(
    name_prefix="wig3",
    n_periods=19,                # Creates 7 complete periods 
    length_short=0.05,          # 5 cm for short segment
    B_short=1.5,                # 2 Tesla peak field
    ratio=3.0,                  # long = 3 × short
    env=env
)

#to place wigglers in the drifts of some straight cells(12/15 in the full ring), we also define a new drift that is dr_str - total wiggler length
env['d_str_after_adding_wig'] = (env['d_str'] - total_wig_length)
env.new('dr_str_after_adding_wig' , xt.Drift , length = 'd_str_after_adding_wig')
#also half of this dr_str_after_adding_wig so that there are symmetric drifts before and after the wiggler
env['d_str_after_adding_wig_half'] = env['d_str_after_adding_wig'] / 2
env.new('dr_str_after_adding_wig_half' , xt.Drift , length = 'd_str_after_adding_wig_half')


#RF cavity parameters
rf_frequency = 347.83e6  # 347 MHz               # T_rev =  1.15e-06 s , harmonic number h  initially estimated to 400, So f_rf = 400 /  1.15e-06 = 347.83 MHz    
rf_voltage = 3.0e6  # 3 MV
env['rf_voltage'] = rf_voltage
env['rf_frequency'] = rf_frequency

env.new('rf', xt.Cavity, voltage='rf_voltage', frequency='rf_frequency', lag='rf_lag')


#  Arc cell ( 2 dipoles, 4 drifts)
# half QF – drift(dr_arc_after_adding_sext_half) - SF - drift(dr_arc_after_adding_sext_half) - bend - drift - QD - drift(dr_arc_after_adding_sext_half) - SD - drift(dr_arc_after_adding_sext_half) - bend - drift - half QF

arc_cell_all_sext = env.new_line(components=[
    env.place('half_qf_arc'),
    env.place('dr_arc_after_adding_sext_half'), env.place('sf'), env.place('dr_arc_after_adding_sext_half')  , env.place('mb'),
    env.place('dr_arc'),
    env.place('qd_arc'), env.place('dr_arc_after_adding_sext_half'), env.place('sd'), env.place('dr_arc_after_adding_sext_half') , env.place('mb'),
    env.place('dr_arc'), env.place('half_qf_arc')
])


arc_cell_all_sext_before_quad = env.new_line(components=[
    env.place('half_qf_arc'), env.place('dr_arc'),
    env.place('mb'),
    env.place('dr_arc_after_adding_sext_half'), env.place('sd'), env.place('dr_arc_after_adding_sext_half') ,
    env.place('qd_arc'), env.place('dr_arc') , env.place('mb'),
    env.place('dr_arc_after_adding_sext_half'), env.place('sf'), env.place('dr_arc_after_adding_sext_half') , env.place('half_qf_arc')
])


arc_cell_1 = env.new_line(components=[
    env.place('half_qf_arc') ,
    env.place('dr_arc') , env.place('mb') ,
    env.place('dr_arc') ,
    env.place('qd_arc') , env.place('dr_arc') , env.place('mb') ,
    env.place('dr_arc') , env.place('half_qf_arc')
])

arc_cell_SF = env.new_line(components=[
    env.place('half_qf_sup_b') ,
    env.place('dr_arc_after_adding_sext_half') , env.place('sf'), env.place('dr_arc_after_adding_sext_half') ,
    env.place('mb'),
    env.place('dr_arc'),
    env.place('qd_arc'), env.place('dr_arc') , env.place('mb') ,
    env.place('dr_arc') , env.place('half_qf_arc')
])

arc_cell_SF_to_left_of_last_quad = env.new_line(components=[
    env.place('half_qf_arc') , env.place('dr_arc'),
    env.place('mb'),
    env.place('dr_arc'),
    env.place('qd_arc'), env.place('dr_arc') , env.place('mb') ,
    env.place('dr_arc_after_adding_sext_half') , env.place('sf'), env.place('dr_arc_after_adding_sext_half') , env.place('half_qf_sup_b')
])


arc_cell_SD = env.new_line(components=[
    env.place('half_qf_arc') ,
    env.place('dr_arc') , env.place('mb'),
    env.place('dr_arc') ,env.place('qd_arc'), env.place('dr_arc_after_adding_sext_half'), env.place('sd'), env.place('dr_arc_after_adding_sext_half') , env.place('mb'),
    env.place('dr_arc'), env.place('half_qf_arc')
])

arc_cell_SD_to_left = env.new_line(components=[
    env.place('half_qf_arc') ,
    env.place('dr_arc') , env.place('mb'), env.place('dr_arc_after_adding_sext_half'), env.place('sd'), env.place('dr_arc_after_adding_sext_half') ,
    env.place('qd_arc'), env.place('dr_arc'), env.place('mb'),
    env.place('dr_arc'), env.place('half_qf_arc')
])


# suppressor cell 1 (0 dipoles, 2 drifts)
# half QF_sup - drift - QD_sup_b - drift - half QF_arc
suppressor_cell_1 = env.new_line(components=[
    env.place('half_qf_sup') ,
    env.place('dr_sup2') , env.place('mb') , env.place('dr_sup2') ,
    env.place('qd_sup_b') , env.place('dr_sup2') , env.place('mb') , env.place('dr_sup2') ,
    env.place('half_qf_sup_b')
])

#suppressor cell 1 inverted (0 dipoles, 2 drifts)
# half QF_arc - drift - QD_sup_b - drift - half QF_sup
suppressor_cell_1_inv = env.new_line(components=[
    env.place('half_qf_sup_b') ,
    env.place('dr_sup2') , env.place('mb') , env.place('dr_sup2') ,
    env.place('qd_sup_b') , env.place('dr_sup2') , env.place('mb') , env.place('dr_sup2') ,
    env.place('half_qf_sup')
])

# suppressor cell 2 (2 dipoles, 4 drifts)
# half QF_str_a – drift – bend – drift – QD_sup_a - drift - bend - drift - half QF_arc
suppressor_cell_2 = env.new_line(components=[
    
    env.place('dr_sup3') , env.place('mb') ,
    env.place('dr_sup2') ,
    env.place('qd_sup_a') , env.place('dr_sup1'), env.place('half_qf_sup')
])

#suppressor cell 2 inverted (2 dipoles, 4 drifts)
# half QF_arc – drift – bend – drift – QD_sup_a - drift - bend - drift - half QF_str_a
suppressor_cell_2_inv = env.new_line(components=[
    env.place('half_qf_sup') ,
    env.place('dr_sup1') ,
    env.place('qd_sup_a') , env.place('dr_sup2') , env.place('mb') ,
    env.place('dr_sup3') ,
])


#triplet in straight sections

triplet_cell = env.new_line(components=[
    env.place('QF_str_doublet'),
    env.place('dr_between_quads_triplet'),
    env.place('QD_str_doublet'),
    *new_drifts_with_markers_wig,
    wiggler_section_1.replicate('straight_wig1'),
    *new_drifts_with_markers_wig,
    env.place('QD_str_triplet'),
    env.place('dr_between_quads_triplet'),
    env.place('QF_str_triplet'),
    env.place('dr_between_quads_triplet'),
    env.place('QD_str_triplet'),
    *new_drifts_with_markers,
    env.place('rf'),
    *new_drifts_with_markers,
    env.place('QD_str_triplet'),
    env.place('dr_between_quads_triplet'),
    env.place('half_QF_str_triplet'), 
])
  
  
triplet_cell_inv = env.new_line(components=[
    env.place('half_QF_str_triplet'),
    env.place('dr_between_quads_triplet'),
    env.place('QD_str_triplet'),
    *new_drifts_with_markers_wig,
    wiggler_section_2.replicate('straight_wig2'),
    *new_drifts_with_markers_wig,
    env.place('QD_str_triplet'),
    env.place('dr_between_quads_triplet'),
    env.place('QF_str_triplet'),
    env.place('dr_between_quads_triplet'),
    env.place('QD_str_triplet'),    
    *new_drifts_with_markers_wig,
    wiggler_section_3.replicate('straight_wig3'),
    *new_drifts_with_markers_wig,
    env.place('QD_str_doublet'),
    env.place('dr_between_quads_triplet'),
    env.place('QF_str_doublet')
])


# 1/6th of the ring as a "transfer line"
#  7 full arc cells --- suppressor cell 1(no bends) + suppressor cell 2(3 bends) --- straight cell 1 + straight cell 2 + straight cell 2 start to centre
def make_transfer_line(idx):
    return env.new_line(components=[
        *[arc_cell_1.replicate(f'arc_{idx}_cell_{i}') for i in range(2)],
        *[arc_cell_SD_to_left.replicate(f'arc_{idx}_cell_{i}') for i in range(1)],
        *[arc_cell_1.replicate(f'arc_{idx}_cell_{i}') for i in range(1)],
        *[arc_cell_all_sext_before_quad.replicate(f'arc_{idx}_cell_{i}') for i in range(1)],
        *[arc_cell_1.replicate(f'arc_{idx}_cell_{i}') for i in range(1)],
        *[arc_cell_SF_to_left_of_last_quad.replicate(f'arc_{idx}_cell_{i}') for i in range(1)],
        *[suppressor_cell_1_inv.replicate(f'sup_start_{idx}_{i}') for i in range(1)],
        *[suppressor_cell_2_inv.replicate(f'sup_start_{idx}_{i}') for i in range(1)],
        *[triplet_cell.replicate(f'triplet_{idx}_{i}') for i in range(1)],
    ])

#transfer line inverted
def make_transfer_line_inv(idx):
    return env.new_line(components=[
        *[triplet_cell_inv.replicate(f'triplet_{idx}_{i}') for i in range(1)],
        *[suppressor_cell_2.replicate(f'sup_start_{idx}_{i}') for i in range(1)],
        *[suppressor_cell_1.replicate(f'sup_start_{idx}_{i}') for i in range(1)],
        *[arc_cell_SF.replicate(f'arc_{idx}_cell_{i}') for i in range(1)],
        *[arc_cell_1.replicate(f'arc_{idx}_cell_{i}') for i in range(1)],
        *[arc_cell_all_sext.replicate(f'arc_{idx}_cell_{i}') for i in range(1)],
        *[arc_cell_1.replicate(f'arc_{idx}_cell_{i}') for i in range(1)],
        *[arc_cell_SD.replicate(f'arc_{idx}_cell_{i}') for i in range(1)],
        *[arc_cell_1.replicate(f'arc_{idx}_cell_{i}') for i in range(2)],
    ])

transfer_line_one_sixth_ring = env.new_line(components=[
     make_transfer_line(2),
 ])

transfer_line_one_sixth_ring_inv = env.new_line(components=[
    make_transfer_line_inv(2),])

transfer_line_one_third_ring = env.new_line(components=[
    make_transfer_line_inv(2), make_transfer_line(2)
])

# transfer_line_one_sixth_ring.build_tracker()
# transfer_line_one_sixth_ring.configure_radiation(model='mean')
# transfer_line_one_sixth_ring.compensate_radiation_energy_loss()

transfer_line_one_sixth_ring.survey().plot()

# Final transfer line length
print(f"1/6th ring: {transfer_line_one_sixth_ring.get_length():.5f} m")



# %%
print(len(transfer_line_one_third_ring.elements))
for ele in transfer_line_one_third_ring.elements:
    print(ele)

arc_twiss = arc_cell_1.twiss(method='4d')

arc_twiss_init = arc_twiss.get_twiss_init(at_element=0)
env['KQF_str_doublet'] = 1.4
env['KQD_str_doublet'] = -1.0

env['kqf_sup_b'] = 1.3

tw_transfer = transfer_line_one_sixth_ring.twiss(
    betx=arc_twiss_init.betx,
    bety=arc_twiss_init.bety,
    alfx=arc_twiss_init.alfx,
    alfy=arc_twiss_init.alfy,
    dx=arc_twiss_init.dx,
    dy=arc_twiss_init.dy,
    dpx=arc_twiss_init.dpx,
    dpy=arc_twiss_init.dpy,
    mux=arc_twiss_init.mux,
    muy=arc_twiss_init.muy,
)


pl = tw_transfer.plot()
pl.ylim(left_lo=0,left_hi=20, right_lo=-0.5, right_hi=1,
        lattice_hi=30, lattice_lo=-5)



# print(arc_twiss_init.betx)
# print(arc_twiss_init.bety)
# print(arc_twiss_init.alfx)
# print(arc_twiss_init.alfy)
# print(arc_twiss_init.mux)
# print(arc_twiss_init.muy)
# print(arc_twiss_init.s)

#tw_transfer.to_pandas()[['s','name','betx','bety','alfx','alfy','mux','muy','dx' , 'dy','dpx','dpy']]

# Convert to pandas DataFrame first:
tw_transfer_df = tw_transfer.to_pandas()[['s','name','betx','bety','alfx','alfy','mux','muy','dx' , 'dy','dpx','dpy']]

# Display the full table to play around a little with the matching location
#display(HTML(tw_transfer_df.to_html()))

# Optionally, export to CSV:
#tw_transfer_df.to_csv('twiss_table.csv')


# %%
# matching

opt_disp_TL = transfer_line_one_sixth_ring.match(
    solve=False,
    compute_chromatic_properties=True,
    method='6d',
    betx=arc_twiss_init.betx,
    bety=arc_twiss_init.bety,
    alfx=arc_twiss_init.alfx,
    alfy=arc_twiss_init.alfy,
    dx=arc_twiss_init.dx,
    dy=arc_twiss_init.dy,
    dpx=arc_twiss_init.dpx,
    dpy=arc_twiss_init.dpy,
    mux=arc_twiss_init.mux,
    muy=arc_twiss_init.muy,  # IMPORTANT: initial conditions from the arc
    vary=[
        xt.Vary('kqd_sup_a', step=1e-4),
        xt.Vary('kqd_sup_b', step=1e-4),
        xt.Vary('KQF_str_doublet', step=1e-4),
        xt.Vary('KQD_str_doublet', step=1e-4),
        xt.Vary('kqf_sup_b', step=1e-4),
        xt.Vary('kqf_sup',  step=1e-4),     
        xt.Vary('KQF_str_triplet', step=1e-4),
        xt.Vary('KQD_str_triplet', step=1e-4),
    ],
    targets=[
        xt.Target(dx=0.0, at='QF_str_doublet.triplet_2_0', tol=1e-6),                      
        xt.Target(dpx=0.0, at='QF_str_doublet.triplet_2_0', tol=1e-6),                     
        xt.Target(alfx=0.0, at='_end_point', tol=1e-5),
        xt.Target(alfy=0.0, at='_end_point', tol=1e-5),
        xt.Target(bety=3.5, at='wig1_short_12.straight_wig1.triplet_2_0', tol=1e-5),  
        xt.Target(betx=3.5, at='wig1_short_7.straight_wig1.triplet_2_0', tol=1e-5),
        xt.Target(bety=3.5, at='marker_str_triplet_70pct.triplet_2_0::1', tol=1e-5),
        xt.Target(betx=3.5, at='marker_str_triplet_120pct.triplet_2_0::1', tol=1e-5),
    ]
)

opt_disp_TL.target_status()
opt_disp_TL.run_jacobian(30)
opt_disp_TL.target_status()
opt_disp_TL.vary_status()



# Optics inspection
tw_TL = transfer_line_one_sixth_ring.twiss(betx=arc_twiss_init.betx,
    bety=arc_twiss_init.bety,
    alfx=arc_twiss_init.alfx,
    alfy=arc_twiss_init.alfy,
    dx=arc_twiss_init.dx,
    dy=arc_twiss_init.dy,
    dpx=arc_twiss_init.dpx,
    dpy=arc_twiss_init.dpy,
    mux=arc_twiss_init.mux,
    muy=arc_twiss_init.muy,)
pl = tw_TL.plot()
pl.ylim(left_hi=25, right_lo=-0.5, right_hi=1.0,
        lattice_hi=1.5, lattice_lo=-7)



# %%

# full ring assembly
RING = env.new_line(components=[
    transfer_line_one_sixth_ring_inv.replicate ('TL1'),
    transfer_line_one_sixth_ring.replicate ('TL2'),
    transfer_line_one_sixth_ring_inv.replicate ('TL3'),
    transfer_line_one_sixth_ring.replicate ('TL4'),
    transfer_line_one_sixth_ring_inv.replicate ('TL5'),
    transfer_line_one_sixth_ring.replicate ('TL6'),
])

RING.replace_all_replicas()
RING.replace_all_repeated_elements()


RING.build_tracker()
RING.configure_radiation(model='mean')
RING.compensate_radiation_energy_loss()

#visualize
RING.survey().plot()
#  Final ring length
print(f"Total circumference of ring: {RING.get_length():.5f} m")

twiss_RING = RING.twiss( method='6d' , eneloss_and_damping=True, radiation_integrals=True, compute_chromatic_properties=True)
twiss_ring_plot = twiss_RING.plot()
twiss_ring_plot.ylim(left_lo=0,left_hi=25, right_lo=-0.2, right_hi=1,
        lattice_hi=0.5, lattice_lo=-10)

twiss_RING.to_pandas()[['s','name','betx','bety','alfx','alfy','mux','muy','dx' , 'dy','dpx','dpy']]

#Convert to pandas DataFrame first:
#twiss_RING_df = twiss_RING.to_pandas()[['s','name','betx','bety','alfx','alfy','mux','muy','dx' , 'dy','dpx','dpy']]




# %%
opt_chroma = RING.match(
    solve=False,
    method='6d',
    vary=[
        xt.Vary('k2_sf', step=1e-3),
        xt.Vary('k2_sd', step=1e-3)
        
        
    ],
    targets=[
        xt.Target(dqx=0.0, tol=1e-6),
        xt.Target(dqy=0.0, tol=1e-6),
    ]
)
opt_chroma.target_status()
opt_chroma.run_jacobian(30)
opt_chroma.target_status()
opt_chroma.vary_status()

# %%
# #export as json
RING.to_json('90lattice_v2o1.json')

# %%



