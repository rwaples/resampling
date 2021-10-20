"""The program runs experiments for different resampling methods for sites and samples.
Pop gen value to estimate: fst
Population: one population
Resampling methods:
1. Bootstrap over sites and samples
2. Jackknife delete one over sites and samples
2. Jackknife delete mj over sites

Last Updated Date: Oct 11, 2021
"""
import pandas as pd
import numpy as np
import sys
import os
import simulation as sim
import observation as obs
from datetime import datetime
from function import get_fst, calculate_ints, location, columns


def experiment(num_exp, num_obs, confidence=0.95, pop_ind=200,
               split_time=50, seq_len=1e9, rec_rate=1e-8, mut_rate=1e-8, exp_seed=None):
    """Run experiment for resampling of site diversity
    num_exp -- number of experiment to run
    num_obs -- number of observations for each num_ind and max_sites
    confidence -- confidence level
    pop_ind -- the population size of each population,
        also the size of the ancestral population
    split_time -- current populations split from the
        ancestral population this many generations ago
    seq_len -- length of the genome, units ~ base-pairs
    rec_rate -- recombination rate, units = rate per bp, per generation
    mut_rate -- mutation rate, units = rate per bp, per generation
    seed -- set seed for the experiment
    """

    result_fst = []
    n_block = int(seq_len // 5e6)
    np.random.seed(exp_seed)
    # generate seeds for population ts
    ts_seeds = np.random.randint(0, 2 ** 32 - 1, num_exp)

    for exp in range(num_exp):
        start = datetime.now()
        print(f'Experiment {exp + 1} starts at:', start)

        pop_ts = sim.sim_population(
            pop_ind=pop_ind,
            split_time=split_time,
            seq_len=seq_len,
            rec_rate=rec_rate,
            mut_rate=mut_rate,
            seed=ts_seeds[exp]
        )

        pop_ts_fst = get_fst(pop_ts)
        print('     Population num sites is:', pop_ts.num_sites)
        print('     Population site fst:', pop_ts_fst)

        num_ind = [20, 50, 100]
        max_sites = [5000, 20000, 50000]

        # generate seeds for all possible observation ts
        np.random.seed(ts_seeds[exp])
        obs_seeds = np.random.randint(0, 2 ** 32 - 1, size=(len(num_ind) * len(max_sites), num_obs))

        row = 0
        for obs_ind in num_ind:
            for obs_sites in max_sites:
                print(f'The desired observation is (num_ind, max_sites) is: {obs_ind, obs_sites}')
                position = np.zeros((num_obs, 15), dtype='int')

                for j in range(num_obs):
                    # make sure the observed value will not surpass the pop value
                    if obs_ind > pop_ts.num_individuals / 2:
                        obs_ind = pop_ts.num_individuals
                    if obs_sites > pop_ts.num_sites:
                        obs_sites = pop_ts.num_sites

                    obs_ts = obs.Fst(pop_ts, obs_ind, obs_sites, seed=obs_seeds[row][j])

                    # confidence intervals of resampling over sites for fst
                    intervals_sites_fst = calculate_ints([obs_ts.bootstrap_sites_fst(),
                                                          obs_ts.jackknife_one_sites_fst(),
                                                          obs_ts.jackknife_mj_sites_fst(n_block)],
                                                         confidence, obs_ts.fst)

                    # confidence intervals of resampling over individuals for fst
                    intervals_ind_fst = calculate_ints([obs_ts.bootstrap_ind_fst(),
                                                        obs_ts.jackknife_one_ind_fst()],
                                                       confidence, obs_ts.fst)

                    # record the position pop value is relative to the intervals
                    position[j] = location(np.concatenate([intervals_sites_fst, intervals_ind_fst]),
                                           pop_ts_fst)

                result_fst.append([exp_seed, (exp + 1), ts_seeds[exp],
                                   seq_len, rec_rate, mut_rate,
                                   pop_ind, pop_ts.num_sites, pop_ts_fst,
                                   obs_ind, max_sites[row % 3], obs_sites, num_obs]
                                  + list(position.sum(0)))

                print('     Fst coverage rate:', result_fst[-1][13:], '\n')

                row += 1

        print(f'The time that the experiment {exp + 1} runs :', datetime.now() - start)

    # save the results to csv file
    result_df = pd.DataFrame(result_fst)
    result_df.columns = columns
    return result_df


if __name__ == '__main__':

    pop_ind = [200, 1000, 5000]
    seq_len = ['1e8', '5e8', '1e9']

    num_exp = int(sys.argv[1])
    num_obs = int(sys.argv[2])
    seed = None
    if len(sys.argv) == 4:
        seed = int(sys.argv[3])

    print('\nNumber of experiment is:', num_exp)
    print('Number of observation in each experiment is:', num_obs)
    print('The seed is:', seed)

    for p in pop_ind:
        for s in seq_len:
            print(f'\ndiploid_size: {p} seq_len: {s} \n')
            prefix = str(p) + "_" + s
            ending = datetime.now().strftime("%m%d")
            fst_df = experiment(num_exp=num_exp, num_obs=num_obs, pop_ind=p, seq_len=float(s),
                                exp_seed=seed)

            if os.path.isdir('../data'):
                fst_df.to_csv(f'../data/{prefix}_fst_{ending}.csv', index=False)
            else:
                fst_df.to_csv(f'/home/users/waplesr/resampling/data/{prefix}_fst_{ending}.csv', index=False)

            print(f'wrote results file: */data/{prefix}_fst_{ending}.csv')


