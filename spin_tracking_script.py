# %%
import xtrack as xt

RING = xt.Line.from_json('lattice_v1o5.json')

# %%
import xobjects as xo

RING.discard_tracker()
RING.build_tracker(_context=xo.ContextCpu())
tw_s = RING.twiss4d(spin=True, polarization=True)


# %%
print(f'Equilibrium polarization: {tw_s.spin_polarization_eq*100:.3f} %')
print(f'Inf. polarization (no depol.): {tw_s.spin_polarization_inf_no_depol*100:.3f} %')
print(f'build up time: {tw_s.spin_t_pol_buildup_s:.3f} s')

# %%
import xtrack as xt
import xpart as xp


RING.configure_radiation(model=None)

tt = RING.get_table()
tt_wig = tt.rows['wig.*']


RING['on_wig'] = 1.0
for nn in tt_wig.name:
    elem = RING[nn]
    if hasattr(elem, 'k0'):
        elem.k0 = f'{elem.k0} * on_wig'

tw_on = RING.twiss(polarization=True)

RING['on_wig'] = 0.0
tw_off = RING.twiss(polarization=True)
RING['on_wig'] = 1.0

tw_spin = RING.twiss(polarization=True)

RING.discard_tracker()
RING.build_tracker(_context=xo.ContextCpu())
RING.configure_radiation(model='mean')
tw = RING.twiss(eneloss_and_damping=True)

import numpy as np

# Generate a matched bunch distribution
np.random.seed(0)
particles = xp.generate_matched_gaussian_bunch(
    line=RING,
    nemitt_x=tw.eq_nemitt_x,
    nemitt_y=0.01*(tw.eq_nemitt_x),  # Assume 1% coupling for vertical emittance
    sigma_z=np.sqrt(tw.eq_gemitt_zeta * tw.bets0),
    num_particles=300,
    engine='linear')

# Add stable phase
particles.zeta += tw.zeta[0]
particles.delta += tw.delta[0]

# Initialize spin of all particles along n0
particles.spin_x = tw_spin.spin_x[0]
particles.spin_y = tw_spin.spin_y[0]
particles.spin_z = tw_spin.spin_z[0]

RING.configure_spin('auto' )
RING.configure_radiation(model='quantum')

# Enable parallelization
RING.discard_tracker()
RING.build_tracker(_context=xo.ContextCpu(omp_num_threads=10))

# Track
num_turns=10000
RING.track(particles, num_turns=num_turns, turn_by_turn_monitor=True,
           with_progress=10)
mon = RING.record_last_track




# %%
#fit depolarization time
mask_alive = mon.state > 0
pol_x = mon.spin_x.sum(axis=0)/mask_alive.sum(axis=0)
pol_y = mon.spin_y.sum(axis=0)/mask_alive.sum(axis=0)
pol_z = mon.spin_z.sum(axis=0)/mask_alive.sum(axis=0)
pol = np.sqrt(pol_x**2 + pol_y**2 + pol_z**2)

i_start = 500 # Skip a few turns (small initial mismatch)
pol_to_fit = pol[i_start:]/pol[i_start]
 
# Fit depolarization time (linear fit)
from scipy.stats import linregress
turns = np.arange(len(pol_to_fit))
slope, intercept, r_value, p_value, std_err = linregress(turns, pol_to_fit)
# Calculate depolarization time
t_dep_turns = -1 / slope


print(f"Depolarization time (turns) from fit: {t_dep_turns:.1f}")
print(f"Depolarization time (seconds) from fit: {t_dep_turns * tw_spin.T_rev0:.3f} s")

import matplotlib.pyplot as plt

plt.figure()
plt.plot(pol_to_fit-1, label='Tracking')
plt.plot(turns, intercept*np.exp(-turns/t_dep_turns) - 1, label='Fit')
plt.ylabel(r'$P/P_0 - 1$')
plt.xlabel('Turn')
plt.subplots_adjust(left=.2)
plt.legend()

p_inf = tw_spin['spin_polarization_inf_no_depol']
t_pol_turns = tw_spin['spin_t_pol_component_s']/tw_spin.T_rev0

print(f"polarization inf: {p_inf:.6f}")
print(f"polarization turn: {t_pol_turns:.6f}")

p_eq = p_inf * 1 / (1 + t_pol_turns/t_dep_turns)

print(f'Equilibrium polarization: {p_eq*100:.2f} %')




# %%



