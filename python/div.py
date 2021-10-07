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

columns = ['experiment', 'exp_seed',
           'pop_ind', 'pop_sites', 'pop_seed',
           'obs_ind', 'obs_sites', 'num_observation',
           'seq_len', 'rec_rate', 'mut_rate', 'population value',
           'bt_sites_lower', 'bt_sites_within', 'bt_sites_above',
           'jk_one_sites_lower', 'jk_one_sites_within', 'jk_one_sites_above',
           'jk_mj_sites_lower', 'jk_mj_sites_within', 'jk_mj_sites_above',
           'bt_ind_lower', 'bt_ind_within', 'bt_ind_above',
           'jk_ind_lower', 'jk_ind_within', 'jk_ind_above']

# interval functions
func_ints = [ci.bt_standard, ci.jk_delete_one, ci.jk_delete_mj]


def location(intervals, pop_val):
    """check whether the population values is with in, lower or above the confidence interval
    return 1 if True else 0
    @intervals: a list of intervals from bt, jk_one, and jk_mj
    @pop_val: float, population value (pop_ts_diversity)
    """
    result = np.zeros(len(intervals) * 3, dtype=int)

    for j, interval in enumerate(intervals):
        if pop_val < interval[0]:
            result[j * 3] += 1
        elif interval[0] <= pop_val <= interval[1]:
            result[j * 3 + 1] += 1
        else:
            result[j * 3 + 2] += 1

    return result


def get_hetero(ts):
    geno = ts.genotype_matrix()
    ac = ((geno[:, ::2] + geno[:, 1::2]) == 1).sum(0)
    return np.mean(ac / ts.num_sites)


def calculate_ints(resample_values, confidence, obs_value):
    """calculate intervals for the resample_values
    @resample_values: bt, jk_one, jk_mj
    @confidence: confidence leve
    @obs_value: observed value
    """
    return [func_ints[i](resample_values[i], confidence, obs_value) for i in range(len(resample_values))]


def experiment(num_exp, num_obs, confidence=0.95,
               pop_ind=200, seq_len=1e9, rec_rate=1e-8, mut_rate=1e-8, seed=None):
    """Run experiment for resampling of site diversity
    num_exp -- number of experiment to run
    num_obs -- number of observations for each num_ind and max_sites
    confidence -- confidence level
    pop_ind -- number of individual for each population
    seq_len -- length of the genome, units ~ base-pairs
    rec_rate -- recombination rate, units = rate per bp, per generation
    mut_rate -- mutation rate, units = rate per bp, per generation
    seed -- seed for each experiment (make the experiments duplicable)
    """
    result_div, result_hetero = [], []
    n_block = int(seq_len // 5e6)
    np.random.seed(seed)
    # generate seeds for population ts
    ts_seeds = np.random.randint(0, 2 ** 32 - 1, num_exp)

    for exp in range(num_exp):
        start = datetime.now()
        print(f'\nExperiment {exp} starts at:', start)

        pop_ts = sim.sim_one_population(
            pop_ind=pop_ind,
            seq_len=seq_len,
            rec_rate=rec_rate,
            mut_rate=mut_rate,
            seed=ts_seeds[exp]
        )
        pop_ts_diversity = pop_ts.diversity(span_normalise=False, windows='sites').mean()
        pop_ts_hetero = get_hetero(pop_ts)
        print('Population num sites is:', pop_ts.num_sites)
        print('Population site diversity:', pop_ts_diversity)
        print('Population heterozygosity:', pop_ts_hetero)

        # change the list here if you would like to explore more
        num_ind = [50]  # [50, 100, 150]
        max_sites = np.geomspace(start=n_block * 2, stop=9000, num=5, endpoint=True, dtype='int')

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
                row += 1
                result_div.append([exp, seed,
                                   pop_ind, pop_ts.num_sites, ts_seeds[exp],
                                   obs_ind, obs_sites, num_obs,
                                   seq_len, rec_rate, mut_rate, pop_ts_diversity]
                                  + list(position_div.sum(0)))

                result_hetero.append([exp, seed,
                                      pop_ind, pop_ts.num_sites, ts_seeds[exp],
                                      obs_ind, obs_sites, num_obs,
                                      seq_len, rec_rate, mut_rate, pop_ts_hetero]
                                     + list(position_hetero.sum(0)))

        print(f'Experiment {exp} runs:', datetime.now() - start)
        print('Diversity outcomes:')
        print(result_div[exp][11:])
        print('Heterozygosity outcomes')
        print(result_hetero[exp][11:])

    result_div_df = pd.DataFrame(result_div)
    result_div_df.columns = columns

    result_hetero_df = pd.DataFrame(result_hetero)
    result_hetero_df.columns = columns

    return result_div_df, result_hetero_df


if __name__ == '__main__':

    print('Input the number of individuals for the population, separated by a comma.')
    pop_ind = list(map(int, input().split(',')))
    print('Input the length of population sequence, separated by a comma.')
    seq_len = list(map(float, input().split(',')))
    print(pop_ind, seq_len)
    print('Do you want to set a seed? (y/n)')
    ans = input()
    seed = None

    if ans == 'y':
        print('Input the seed number')
        seed = int(input())
    print(f'base seed : {seed}')

    print('Input the number of experiments you want to run')
    num_exp = int(input())
    print('Input  the number of observation you want to have for each experiment')
    num_obs = int(input())
    print()

    for p in pop_ind:
        for s in seq_len:
            print(f' diploid_size: {p} seq_len: {s}')
            prefix = datetime.now().strftime("%m%d%H%M%S")
            div_df, hetero_df = experiment(num_exp=num_exp, num_obs=num_obs, pop_ind=p, seq_len=s, seed=seed)
            div_df.to_csv(f'../data/{prefix}_site_diversity.csv', index=False)
            print(f'wrote results file: ../data/{prefix}_site_diversity.csv')
            hetero_df.to_csv(f'../data/{prefix}_heterozygosity.csv', index=False)
            print(f'wrote results file: ../data/{prefix}_heterozygosity.csv')