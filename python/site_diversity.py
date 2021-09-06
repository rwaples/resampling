"""The program runs experiments for different resampling methods for sites and samples.
Pop gen value to estimate: site diversity
Population: one population
Resampling methods:
1. Bootstrap over sites and samples
2. Jackknife delete one over sites and samples
2. Jackknife delete mj over sites and samples
"""
import pandas as pd
import numpy as np
import intervals as ci
import simulation as sim
from observation import Observation1
from datetime import datetime


def check_within(interval, pop_val):
    """check whether the population values is with in the confidence interval
    return 1 if True else 0
    @interval: a tuple (lower, upper)
    @pop_val: population value (pop_ts_diversity, pop_fst)
    """
    return 1 if interval[0] < pop_val < interval[1] else 0


def experiment(num_exp, num_obs, confidence):
    """Run experiment for resampling of site diversity
    @num_exp: number of experiment to run
    @num_obs: number of observations for each num_inds and max_sites
    @confidence: confidence level
    """
    result = []
    start = datetime.now()
    print('Experiment starts at:', start)

    for exp in range(num_exp):
        print('Experiment:', exp)

        pop_ts = sim.sim_one_population(
            diploid_size=200,
            seq_len=1e9,
            rec_rate=1e-8,
            mut_rate=1e-8
        )
        pop_ts_diversity = pop_ts.diversity(span_normalise=False, windows='sites').mean()
        print('Population site diversity:', pop_ts_diversity)

        num_inds_list = np.linspace(50, 200, 4, dtype=int)
        max_sites_list = np.linspace(1000, 5000, 5, dtype=int)

        for num_inds in num_inds_list:
            for max_sites in max_sites_list:
                print(f'The shape of observed population (num_inds, max_sites) is: {num_inds, max_sites}')
                resample_start = datetime.now()
                print('Resample starts:', resample_start)

                within = np.zeros((num_obs, 6))

                for j in range(num_obs):
                    obs_ts = Observation1(pop_ts, num_inds, max_sites)
                    obs_ts_diversity = np.mean(obs_ts.site_diversity)

                    # confidence intervals bound when resampling over sites
                    bt_sites = ci.bt_standard(obs_ts.bootstrap_sites_diversity(),
                                              confidence, obs_ts_diversity)
                    jk_delete_one_sites = ci.jk_delete_one(obs_ts.jackknife_one_sites_diversity(),
                                                           confidence, obs_ts_diversity)
                    jk_delete_mj_sites = ci.jk_delete_mj(obs_ts.jackknife_mj_sites_diversity(),
                                                         confidence, obs_ts_diversity)

                    # confidence intervals when resampling over samples
                    bt_samples = ci.bt_standard(obs_ts.bootstrap_samples_diversity(),
                                                confidence, obs_ts_diversity)
                    jk_delete_one_samples = ci.jk_delete_one(obs_ts.jackknife_one_samples_diversity(),
                                                             confidence, obs_ts_diversity)
                    jk_delete_mj_samples = ci.jk_delete_mj(obs_ts.jackknife_mj_samples_diversity(),
                                                           confidence, obs_ts_diversity)

                    resample_values = [bt_sites, jk_delete_one_sites, jk_delete_mj_sites,
                                       bt_samples, jk_delete_one_samples, jk_delete_mj_samples]

                    within[j] = list(map(check_within, resample_values, [pop_ts_diversity] * 6))

                coverage = np.mean(within, axis=0)
                print('The coverage rate for each method is', coverage)
                print('Resample run time:', datetime.now() - resample_start, '\n')
                result.append([exp, num_inds, max_sites, pop_ts_diversity] + list(coverage))

    # save the results to csv file
    result_df = pd.DataFrame(result)
    result_df.columns = ['experiment', 'num_inds', 'max_sites', 'pop_ts_diversity',
                         'bt_sites', 'jk_delete_one_sites', 'jk_delete_mj_sites',
                         'bt_samples', 'jk_delete_one_samples', 'jk_delete_mj_samples']
    print('Experiment run time:', datetime.now() - start)
    return result_df


if __name__ == '__main__':
    print(experiment(num_exp=1, num_obs=1, confidence=0.95))
