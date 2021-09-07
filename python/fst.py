"""The program runs experiments for different resampling methods for sites and samples.
Pop gen value to estimate: fst
Population: one population
Resampling methods:
1. Bootstrap over sites and samples
2. Jackknife delete one over sites and samples
2. Jackknife delete mj over sites
"""
import pandas as pd
import numpy as np
import allel
import intervals as ci
import simulation as sim
import observation as obs
from datetime import datetime


def check_within(interval, pop_val):
    """check whether the population values is with in the confidence interval
    return 1 if True else 0
    @interval: a tuple (lower, upper)
    @pop_val: population value (pop_ts_diversity, pop_fst)
    """
    return 1 if interval[0] < pop_val < interval[1] else 0


def get_fst(ts):
    """returns Hudson's Fst
    @ts = tree sequence
    """
    popA_samples = ts.samples(population=0)
    popB_samples = ts.samples(population=1)
    sites_index = np.arange(len(ts.sites()))
    ga = allel.GenotypeArray(
        ts.genotype_matrix().reshape(
            ts.num_sites, ts.num_samples, 1),
        dtype='i1')
    # count alleles within each population at the selected sites and inds
    ac1 = ga[sites_index][:, popA_samples, :].count_alleles()
    ac2 = ga[sites_index][:, popB_samples, :].count_alleles()
    # calculate Hudson's Fst (weighted)
    num, denom = allel.hudson_fst(ac1, ac2)
    fst = np.sum(num) / np.sum(denom)
    return fst


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

        pop_ts = sim.sim_population(
            diploid_size=200,
            split_time=50,
            seq_len=1e9,
            rec_rate=1e-8,
            mut_rate=1e-8
        )
        pop_ts_fst = get_fst(pop_ts)
        print('Population site fst:', pop_ts_fst)

        # num_inds_list = np.linspace(50, 200, 4, dtype=int)
        # max_sites_list = np.linspace(1000, 5000, 5, dtype=int)
        num_inds_list = [50]
        max_sites_list = [1000]

        for num_inds in num_inds_list:
            for max_sites in max_sites_list:
                print(f'The shape of observed population (num_inds, max_sites) is: {num_inds, max_sites}')
                resample_start = datetime.now()
                print('Resample starts:', resample_start)

                within = np.zeros((num_obs, 5))

                for j in range(num_obs):
                    obs_ts = obs.Observation2(pop_ts, num_inds, max_sites)
                    obs_ts_fst = obs_ts.fst

                    # confidence intervals bound when resampling over sites
                    bt_sites = ci.bt_standard(obs_ts.bootstrap_sites_fst(),
                                              confidence, obs_ts_fst)
                    jk_delete_one_sites = ci.jk_delete_one(obs_ts.jackknife_one_sites_fst(),
                                                           confidence, obs_ts_fst)
                    jk_delete_mj_sites = ci.jk_delete_mj(obs_ts.jackknife_mj_sites_fst(),
                                                         confidence, obs_ts_fst)

                    # confidence intervals when resampling over samples
                    bt_samples = ci.bt_standard(obs_ts.bootstrap_samples_fst(),
                                                confidence, obs_ts_fst)
                    jk_samples = ci.jk_delete_one(obs_ts.jackknife_samples_fst(),
                                                  confidence, obs_ts_fst)

                    resample_values = [bt_sites, jk_delete_one_sites, jk_delete_mj_sites,
                                       bt_samples, jk_samples]

                    within[j] = list(map(check_within, resample_values, [pop_ts_fst] * 6))

                coverage = np.mean(within, axis=0)
                print('The coverage rate for each method is', coverage)
                print('Resample run time:', datetime.now() - resample_start, '\n')
                result.append([exp, num_inds, max_sites, pop_ts_fst] + list(coverage))

    # save the results to csv file
    result_df = pd.DataFrame(result)
    result_df.columns = ['experiment', 'num_inds', 'max_sites', 'pop_ts_fst',
                         'bt_sites', 'jk_delete_one_sites', 'jk_delete_mj_sites',
                         'bt_samples', 'jk_samples']
    print('Experiment run time:', datetime.now() - start)
    return result_df


if __name__ == '__main__':
    print(experiment(num_exp=1, num_obs=20, confidence=0.95))
