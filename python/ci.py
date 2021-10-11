"""This file is used for testing purpose.

Last Updated Date: Oct 11, 2021
"""
import numpy as np
import simulation as sim
import observation as obs
from div import get_hetero, calculate_ints, location

if __name__ == '__main__':
    pop_ts = sim.sim_one_population(
        pop_ind=100,
        seq_len=1e8,
        rec_rate=1e-8,
        mut_rate=1e-8,
        seed=1791095845
    )

    pop_ts_diversity = pop_ts.diversity(span_normalise=False, windows='sites').mean()
    pop_ts_hetero = get_hetero(pop_ts)
    print('Population num sites is:', pop_ts.num_sites)
    print('Population site diversity:', pop_ts_diversity)
    print('Population heterozygosity:', pop_ts_hetero)

    np.random.seed(1791095845)
    obs_seeds = np.random.randint(0, 2 ** 32 - 1, size=(5, 10))[2, :]

    num_ind = 50
    max_sites = 600
    num_obs = 10
    confidence = 0.95
    n_block = int(1e8 // 5e6)
    intervals_sites_diversity = [[]] * 10
    intervals_ind_diversity = [[]] * 10
    intervals_sites_hetero = [[]] * 10
    intervals_ind_hetero = [[]] * 10

    position_div = np.zeros((num_obs, 15), dtype=int)
    position_hetero = np.zeros((num_obs, 15), dtype=int)

    for i in range(num_obs):
        obs_ts = obs.Div(pop_ts, num_ind, max_sites, seed=obs_seeds[i])
        obs_ts_diversity = obs_ts.div
        obs_ts_hetero = obs_ts.hetero

        # confidence intervals of resampling over sites for diversity
        intervals_sites_diversity[i] = calculate_ints([obs_ts.bootstrap_sites_diversity(),
                                                       obs_ts.jackknife_one_sites_diversity(),
                                                       obs_ts.jackknife_mj_sites_diversity(n_block)],
                                                      confidence, obs_ts_diversity)

        # confidence intervals of resampling over individuals for diversity
        intervals_ind_diversity[i] = calculate_ints([obs_ts.bootstrap_ind_diversity(),
                                                     obs_ts.jackknife_one_ind_diversity()],
                                                    confidence, obs_ts_diversity)

        # confidence intervals of resampling over sites for hetero
        intervals_sites_hetero[i] = calculate_ints([obs_ts.bootstrap_sites_hetero(),
                                                    obs_ts.jackknife_one_sites_hetero(),
                                                    obs_ts.jackknife_mj_sites_hetero(n_block)],
                                                   confidence, obs_ts_hetero)

        # confidence intervals of resampling over individuals for hetero
        intervals_ind_hetero[i] = calculate_ints([obs_ts.bootstrap_ind_hetero(),
                                                  obs_ts.jackknife_one_ind_hetero()],
                                                 confidence, obs_ts_hetero)

        position_div[i] = location(np.concatenate([intervals_sites_diversity[i], intervals_ind_diversity[i]]),
                                   pop_ts_diversity)

        position_hetero[i] = location(np.concatenate([intervals_sites_hetero[i], intervals_ind_hetero[i]]),
                                      pop_ts_hetero)

    print('Resample over sites: diversity')
    print(np.array(intervals_sites_diversity), '\n')
    print('Resample over individual: diversity')
    print(np.array(intervals_ind_diversity), '\n')
    print('Resample over sites: hetero')
    print(np.array(intervals_sites_hetero), '\n')
    print('Resample over individual: hetero')
    print(np.array(intervals_ind_hetero), '\n')
    print(position_div.sum(0), '\n')
    print(position_hetero.sum(0), '\n')
