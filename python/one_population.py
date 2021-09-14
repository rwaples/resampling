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

columns = ['experiment', 'num_ind', 'max_sites', 'num_observation',
           'seq_len', 'rec_rate', 'mut_rate', 'population value',
           'bt_sites_lower', 'bt_sites_within', 'bt_sites_above',
           'jk_one_sites_lower', 'jk_one_sites_within', 'jk_one_sites_above',
           'jk_mj_sites_lower', 'jk_mj_sites_within', 'jk_mj_sites_above',
           'bt_ind_lower', 'bt_ind_within', 'bt_ind_above',
           'jk_ind_lower', 'jk_ind_within', 'jk_ind_above']


def check(intervals, pop_val):
    """check whether the population values is with in, lower or above the confidence interval
    return 1 if True else 0
    @intervals: a tuple (lower, upper)
    @pop_val: population value (pop_ts_diversity, pop_fst)
    """
    result = np.zeros(len(intervals[0]) * 3)
    for i in range(len(intervals)):
        for j, interval in enumerate(intervals[i]):
            if pop_val < interval[0]:
                result[j * 3] += 1
            elif interval[0] <= pop_val <= interval[1]:
                result[j * 3 + 1] += 1
            else:
                result[j * 3 + 2] += 1

    return list(result)


def get_hetero(ts):
    geno = ts.genotype_matrix()
    ac = ((geno[:, ::2] + geno[:, 1::2]) == 1).sum(0)
    return np.mean(ac / ts.num_sites)


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
    result_div, result_hetero = [], []
    n_block = int(seq_len // 5e6)
    start = datetime.now()
    print('Experiment starts at:', start)

    for exp in range(num_exp):
        print('Experiment:', exp)

        pop_ts = sim.sim_one_population(
            diploid_size=diploid_size,
            seq_len=seq_len,
            rec_rate=rec_rate,
            mut_rate=mut_rate,
            seed=exp + 1
        )
        pop_ts_diversity = pop_ts.diversity(span_normalise=False, windows='sites').mean()
        pop_ts_hetero = get_hetero(pop_ts)
        print('Population site diversity:', pop_ts_diversity)
        print('Population heterozygosity:', pop_ts_hetero)

        # change the list here if you would like to explore more
        num_ind_list = [50]  # [50, 100, 150]
        max_sites_list = [1000]  # [1000, 2000, 3000, 4000, 5000]
        assert max(num_ind_list) <= pop_ts.num_individuals and max(max_sites_list) <= pop_ts.num_sites

        for num_ind in num_ind_list:
            for max_sites in max_sites_list:
                print(f'The shape of observed population (num_ind, max_sites) is: {num_ind, max_sites}')
                resample_start = datetime.now()
                # print('Resample starts at:', resample_start)

                intervals_div, intervals_hetero = [], []

                for j in range(1, num_obs + 1):
                    obs_ts = obs.Observation1(pop_ts, num_ind, max_sites, seed=j * 5000)
                    obs_ts_diversity = np.mean(obs_ts.site_diversity)
                    obs_ts_hetero = np.mean(obs_ts.hetero)

                    # confidence intervals of resampling over sites for diversity
                    bt_sites_diversity = ci.bt_standard(obs_ts.bootstrap_sites_diversity(),
                                                        confidence, obs_ts_diversity)
                    jk_one_sites_diversity = ci.jk_delete_one(obs_ts.jackknife_one_sites_diversity(),
                                                              confidence, obs_ts_diversity)
                    jk_mj_sites_diversity = ci.jk_delete_mj(obs_ts.jackknife_mj_sites_diversity(n_block),
                                                            confidence, obs_ts_diversity)

                    # confidence intervals of resampling over individuals for diversity
                    bt_ind_diversity = ci.bt_standard(obs_ts.bootstrap_ind_diversity(),
                                                      confidence, obs_ts_diversity)
                    jk_ind_diversity = ci.jk_delete_one(obs_ts.jackknife_one_ind_diversity(),
                                                        confidence, obs_ts_diversity)

                    intervals_div.append([bt_sites_diversity, jk_one_sites_diversity, jk_mj_sites_diversity,
                                          bt_ind_diversity, jk_ind_diversity])

                    # confidence intervals of resampling over sites for hetero
                    bt_sites_hetero = ci.bt_standard(obs_ts.bootstrap_sites_hetero(),
                                                     confidence, obs_ts_hetero)
                    jk_one_sites_hetero = ci.jk_delete_one(obs_ts.jackknife_one_sites_hetero(),
                                                           confidence, obs_ts_hetero)
                    jk_mj_sites_hetero = ci.jk_delete_mj(obs_ts.jackknife_mj_sites_hetero(n_block),
                                                         confidence, obs_ts_hetero)

                    # confidence intervals of resampling over individuals for hetero
                    bt_ind_hetero = ci.bt_standard(obs_ts.bootstrap_ind_hetero(),
                                                   confidence, obs_ts_hetero)
                    jk_ind_hetero = ci.jk_delete_one(obs_ts.jackknife_one_ind_hetero(),
                                                     confidence, obs_ts_hetero)

                    intervals_hetero.append([bt_sites_hetero, jk_one_sites_hetero, jk_mj_sites_hetero,
                                             bt_ind_hetero, jk_ind_hetero])

                print('Resample run time:', datetime.now() - resample_start, )
                loc_div = check(intervals_div, pop_ts_diversity)
                loc_hetero = check(intervals_hetero, pop_ts_hetero)
                print(loc_div)
                print(loc_hetero)
                result_div.append([exp, num_ind, max_sites, num_obs,
                                   seq_len, rec_rate, mut_rate, pop_ts_diversity]
                                  + loc_div)

                result_hetero.append([exp, num_ind, max_sites, num_obs,
                                      seq_len, rec_rate, mut_rate, pop_ts_hetero]
                                     + loc_hetero)

    print('Experiment run time:', datetime.now() - start, '\n')
    result_div_df = pd.DataFrame(result_div)
    result_div_df.columns = columns

    result_hetero_df = pd.DataFrame(result_hetero)
    result_hetero_df.columns = columns

    return result_div_df, result_hetero_df


if __name__ == '__main__':
    prefix = datetime.now().strftime("%m%d")
    div_df, hetero_df = experiment(num_exp=1, num_obs=100, confidence=0.95)
    div_df.to_csv(f'../data/{prefix}_site_diversity.csv', index=False)
    hetero_df.to_csv(f'../data/{prefix}_heterozygosity.csv', index=False)
