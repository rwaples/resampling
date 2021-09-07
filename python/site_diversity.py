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
import observation as obs
from datetime import datetime


def check(intervals, pop_val):
    """check whether the population values is with in, lower or above the confidence interval
    return 1 if True else 0
    @intervals: a tuple (lower, upper)
    @pop_val: population value (pop_ts_diversity, pop_fst)
    """
    result = np.zeros(18)
    for i in range(len(intervals)):
        for j, interval in enumerate(intervals[i]):
            if pop_val < interval[0]:
                result[j * 3] += 1
            elif interval[0] <= pop_val <= interval[1]:
                result[j * 3 + 1] += 1
            else:
                result[j * 3 + 2] += 1

    return list(result)


def experiment(num_exp, num_obs, confidence=0.95,
               diploid_size=200, seq_len=1e9, rec_rate=1e-8, mut_rate=1e-8):
    """Run experiment for resampling of site diversity
    @num_exp: number of experiment to run
    @num_obs: number of observations for each num_ind and max_sites
    @confidence: confidence level
    @diploid_size = the population size of each population,
        also the size of the ancestral population
    @seq_len = length of the genome, units ~ base-pairs
    @rec_rate = recombination rate, units = rate per bp, per generation
    @mut_rate = mutation rate, units = rate per bp, per generation
    """
    result = []
    start = datetime.now()
    print('Experiment starts at:', start)

    for exp in range(num_exp):
        print('Experiment:', exp)

        pop_ts = sim.sim_one_population(
            diploid_size=diploid_size,
            seq_len=seq_len,
            rec_rate=rec_rate,
            mut_rate=mut_rate
        )
        pop_ts_diversity = pop_ts.diversity(span_normalise=False, windows='sites').mean()
        print('Population site diversity:', pop_ts_diversity)

        # change the list here if you would like to explore more
        num_ind_list = [50, 100, 150]
        max_sites_list = [1000, 2000, 3000, 4000, 5000]
        assert max(num_ind_list) <= pop_ts.num_individuals and max(max_sites_list) <= pop_ts.num_sites

        for num_ind in num_ind_list:
            for max_sites in max_sites_list:
                print(f'The shape of observed population (num_ind, max_sites) is: {num_ind, max_sites}')
                resample_start = datetime.now()
                # print('Resample starts at:', resample_start)

                intervals = []

                for j in range(num_obs):
                    obs_ts = obs.Observation1(pop_ts, num_ind, max_sites)
                    obs_ts_diversity = np.mean(obs_ts.site_diversity)

                    # confidence intervals bound when resampling over sites
                    bt_sites = ci.bt_standard(obs_ts.bootstrap_sites_diversity(),
                                              confidence, obs_ts_diversity)
                    jk_one_sites = ci.jk_delete_one(obs_ts.jackknife_one_sites_diversity(),
                                                    confidence, obs_ts_diversity)
                    jk_mj_sites = ci.jk_delete_mj(obs_ts.jackknife_mj_sites_diversity(),
                                                  confidence, obs_ts_diversity)

                    # confidence intervals when resampling over samples
                    bt_ind = ci.bt_standard(obs_ts.bootstrap_ind_diversity(),
                                            confidence, obs_ts_diversity)
                    jk_one_ind = ci.jk_delete_one(obs_ts.jackknife_one_ind_diversity(),
                                                  confidence, obs_ts_diversity)
                    jk_mj_ind = ci.jk_delete_mj(obs_ts.jackknife_mj_ind_diversity(),
                                                confidence, obs_ts_diversity)

                    intervals.append([bt_sites, jk_one_sites, jk_mj_sites,
                                      bt_ind, jk_one_ind, jk_mj_ind])

                print('Resample run time:', datetime.now() - resample_start,)
                # where the pop_ts_diversity is located relative to the confidence interval
                location = check(intervals, pop_ts_diversity)
                # print(location)
                result.append([exp, num_ind, max_sites, num_obs,
                               seq_len, rec_rate, mut_rate, pop_ts_diversity]
                              + location)

    print('Experiment run time:', datetime.now() - start, '\n')
    result_df = pd.DataFrame(result)
    result_df.columns = ['experiment', 'num_ind', 'max_sites', 'num_observation',
                         'seq_len', 'rec_rate', 'mut_rate', 'pop_ts_diversity',
                         'bt_sites_lower', 'bt_sites_within', 'bt_sites_above',
                         'jk_one_sites_lower', 'jk_one_sites_within', 'jk_one_sites_above',
                         'jk_mj_sites_lower', 'jk_mj_sites_within', 'jk_mj_sites_above',
                         'bt_ind_lower', 'bt_ind_within', 'bt_ind_above',
                         'jk_one_ind_lower', 'jk_one_ind_within', 'jk_one_ind_above',
                         'jk_mj_ind_lower', 'jk_mj_ind_within', 'jk_mj_ind_above']

    return result_df


if __name__ == '__main__':
    prefix = datetime.now().strftime("%m%d")
    df = experiment(num_exp=1, num_obs=100, confidence=0.95)
    df.to_csv(f'../data/{prefix}_site_diversity.csv', index=False)
