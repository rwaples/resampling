"""The program runs experiments for different resampling methods for sites and samples.
Pop gen value to estimate: site diversity
Population: one population
Resampling methods:
1. Bootstrap over sites and samples
2. Jackknife delete one over sites and samples
2. Jackknife delete mj over sites and samples

Last Updated Date: Oct 11, 2021
"""
import pandas as pd
import numpy as np
import simulation as sim
import observation as obs
import sys
import os
from datetime import datetime
from function import get_hetero, calculate_ints, location, columns


def experiment(num_exp, num_obs, confidence=0.95,
               pop_ind=200, seq_len=1e9, rec_rate=1e-8, mut_rate=1e-8, exp_seed=None):
    """Run experiment for resampling of site diversity
    num_exp -- number of experiment to run
    num_obs -- number of observations for each num_ind and max_sites
    confidence -- confidence level
    pop_ind -- number of individual for each population
    seq_len -- length of the genome, units ~ base-pairs
    rec_rate -- recombination rate, units = rate per bp, per generation
    mut_rate -- mutation rate, units = rate per bp, per generation
    exp_seed -- seed for the overall experiment (make the experiments duplicable)
    """

    result_div, result_hetero = [], []
    n_block = int(seq_len // 5e6)
    np.random.seed(exp_seed)
    # generate seeds for population ts
    ts_seeds = np.random.randint(0, 2 ** 32 - 1, num_exp)

    for exp in range(num_exp):
        start = datetime.now()
        print(f'Experiment {exp + 1} starts at:', start)

        pop_ts = sim.sim_one_population(
            pop_ind=pop_ind,
            seq_len=seq_len,
            rec_rate=rec_rate,
            mut_rate=mut_rate,
            seed=ts_seeds[exp]
        )
        pop_ts_diversity = pop_ts.diversity(span_normalise=False, windows='sites').mean()
        pop_ts_hetero = get_hetero(pop_ts)
        print('     Population num sites is:', pop_ts.num_sites)
        print('     Population site diversity:', pop_ts_diversity)
        print('     Population heterozygosity:', pop_ts_hetero)

        num_ind = [20, 50, 100]
        max_sites = [5000, 20000, 50000]

        # generate seeds for all possible observation ts
        np.random.seed(ts_seeds[exp])
        obs_seeds = np.random.randint(0, 2 ** 32 - 1, size=(len(num_ind) * len(max_sites), num_obs))

        # row number of obs_seeds
        row = 0
        for obs_ind in num_ind:
            for obs_sites in max_sites:
                print(f'The desired observation is (num_ind, max_sites) is: {obs_ind, obs_sites}')
                position_div = np.zeros((num_obs, 15), dtype=int)
                position_hetero = np.zeros((num_obs, 15), dtype=int)

                for j in range(num_obs):
                    # make sure the observed value will not surpass the pop value
                    if obs_ind > pop_ts.num_individuals:
                        obs_ind = pop_ts.num_individuals
                    if obs_sites > pop_ts.num_sites:
                        obs_sites = pop_ts.num_sites

                    obs_ts = obs.Div(pop_ts, obs_ind, obs_sites, seed=obs_seeds[row][j])
                    obs_ts_diversity = obs_ts.div
                    obs_ts_hetero = obs_ts.hetero

                    # confidence intervals of resampling over sites for diversity
                    intervals_sites_diversity = calculate_ints([obs_ts.bootstrap_sites_diversity(),
                                                                obs_ts.jackknife_one_sites_diversity(),
                                                                obs_ts.jackknife_mj_sites_diversity(n_block)],
                                                               confidence, obs_ts_diversity)

                    # confidence intervals of resampling over individuals for diversity
                    intervals_ind_diversity = calculate_ints([obs_ts.bootstrap_ind_diversity(),
                                                              obs_ts.jackknife_one_ind_diversity()],
                                                             confidence, obs_ts_diversity)

                    # confidence intervals of resampling over sites for hetero
                    intervals_sites_hetero = calculate_ints([obs_ts.bootstrap_sites_hetero(),
                                                             obs_ts.jackknife_one_sites_hetero(),
                                                             obs_ts.jackknife_mj_sites_hetero(n_block)],
                                                            confidence, obs_ts_hetero)

                    # confidence intervals of resampling over individuals for hetero
                    intervals_ind_hetero = calculate_ints([obs_ts.bootstrap_ind_hetero(),
                                                           obs_ts.jackknife_one_ind_hetero()],
                                                          confidence, obs_ts_hetero)

                    position_div[j] = location(np.concatenate([intervals_sites_diversity, intervals_ind_diversity]),
                                               pop_ts_diversity)
                    position_hetero[j] = location(np.concatenate([intervals_sites_hetero, intervals_ind_hetero]),
                                                  pop_ts_hetero)

                # update the row number
                result_div.append([exp_seed, (exp + 1), ts_seeds[exp],
                                   seq_len, rec_rate, mut_rate,
                                   pop_ind, pop_ts.num_sites, pop_ts_diversity,
                                   obs_ind, max_sites[row % 3], obs_sites, num_obs]
                                  + list(position_div.sum(0)))

                result_hetero.append([exp_seed, (exp + 1), ts_seeds[exp],
                                      seq_len, rec_rate, mut_rate,
                                      pop_ind, pop_ts.num_sites, pop_ts_hetero,
                                      obs_ind, max_sites[row % 3], obs_sites, num_obs]
                                     + list(position_hetero.sum(0)))

                print('     Diversity coverage rate:', result_div[-1][13:])
                print('     Heterozygosity coverage rate', result_hetero[-1][13:], '\n')

                row += 1

        print(f'The time that the experiment {exp + 1} runs :', datetime.now() - start)

    result_div_df = pd.DataFrame(result_div)
    result_div_df.columns = columns

    result_hetero_df = pd.DataFrame(result_hetero)
    result_hetero_df.columns = columns

    return result_div_df, result_hetero_df


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
            div_df, hetero_df = experiment(num_exp=num_exp, num_obs=num_obs, pop_ind=p, seq_len=float(s),
                                           exp_seed=seed)
            if os.path.isdir('../data'):
                div_df.to_csv(f'../data/{prefix}_site_diversity_{ending}.csv', index=False)
                print(f'wrote results file: ../data/{prefix}_site_diversity_{ending}.csv')
                hetero_df.to_csv(f'../data/{prefix}_heterozygosity_{ending}.csv', index=False)
                print(f'wrote results file: ../data/{prefix}_heterozygosity_{ending}.csv \n')
            else:
                div_df.to_csv(f'/home/users/waplesr/resampling/data/{prefix}_site_diversity_{ending}.csv', index=False)
                print(f'wrote results file: /home/users/waplesr/resampling/data/{prefix}_site_diversity_{ending}.csv')
                hetero_df.to_csv(f'/home/users/waplesr/resampling/data/{prefix}_heterozygosity_{ending}.csv', index=False)
                print(f'wrote results file: /home/users/waplesr/resampling/data/{prefix}_heterozygosity_{ending}.csv')
