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
               diploid_size=200, seq_len=1e9, rec_rate=1e-8, mut_rate=1e-8, seed=None):
    """Run experiment for resampling of site diversity
    @num_exp: number of experiment to run
    @num_obs: number of observations for each num_ind and max_sites
    @confidence: confidence level
    @diploid_size = the population size of each population,
        also the size of the ancestral population
    @seq_len = length of the genome, units ~ base-pairs
    @rec_rate = recombination rate, units = rate per bp, per generation
    @mut_rate = mutation rate, units = rate per bp, per generation
    @seed = set seed for the experiment
    """
    result_div, result_hetero = [], []
    n_block = int(seq_len // 5e6)
    np.random.seed(seed)
    # generate seeds for population ts
    ts_seeds = np.random.randint(0, 2 ** 32 - 1, num_exp)

    for exp in range(num_exp):
        start = datetime.now()
        print(f'\n Experiment {exp} starts at:', exp)

        pop_ts = sim.sim_one_population(
            diploid_size=diploid_size,
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
        num_ind_list = [50]  # [50, 100, 150]
        max_sites_list = [5000, 20000, 50000]  # [1000, 2000, 3000, 4000, 5000]
        #assert max(num_ind_list) <= pop_ts.num_individuals and max(max_sites_list) <= pop_ts.num_sites, \
        #    "Number of ind and max_sites must be smaller than the population"

        # generate seeds for all possible observation ts
        np.random.seed(ts_seeds[exp])
        obs_seeds = np.random.randint(0, 2 ** 32 - 1, size=(len(num_ind_list) * len(max_sites_list), num_obs))

        # row number of obs_seeds
        row = 0
        for num_ind in num_ind_list:
            for max_sites in max_sites_list:
                print(f'The desired observation is (num_ind, max_sites) is: {num_ind, max_sites}')
                position_div = np.zeros((num_obs, 15), dtype=int)
                position_hetero = np.zeros((num_obs, 15), dtype=int)

                for j in range(num_obs):
                    if max_sites > pop_ts.num_sites:
                        max_sites = pop_ts.num_sites
                    obs_ts = obs.Div(pop_ts, num_ind, max_sites, seed=obs_seeds[row][j])
                    obs_ts_diversity = np.mean(obs_ts.site_diversity)
                    obs_ts_hetero = np.mean(obs_ts.hetero)

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
                result_div.append([exp, num_ind, max_sites, num_obs,
                                   seq_len, rec_rate, mut_rate, pop_ts_diversity]
                                  + list(position_div.sum(0)))

                result_hetero.append([exp, num_ind, max_sites, num_obs,
                                      seq_len, rec_rate, mut_rate, pop_ts_hetero]
                                     + list(position_hetero.sum(0)))

        print(f'Experiment {exp} runs:', datetime.now() - start)
        print('Diversity outcomes:')
        print(result_div[exp][8:])
        print('Heterozygosity outcomes')
        print(result_hetero[exp][8:])

    result_div_df = pd.DataFrame(result_div)
    result_div_df.columns = columns

    result_hetero_df = pd.DataFrame(result_hetero)
    result_hetero_df.columns = columns

    return result_div_df, result_hetero_df


if __name__ == '__main__':
    prefix = datetime.now().strftime("%m%d%H%M")
    print(f'data prefix: {prefix}')
    seed = 1
    print(f'base seed : {seed}')
    #diploid_size = [200, 1000, 1500]
    #seq_len = [1e8, 5e8, 1e9]
    diploid_size = [1000]
    seq_len = [ 1e9]
    for i, (d, s) in enumerate(zip(diploid_size, seq_len)):
        print(f'diploid_size: {diploid_size} seq_len: {seq_len}')
        div_df, hetero_df = experiment(num_exp=50, num_obs=10, diploid_size=d, seq_len=s, seed=seed)
        div_df.to_csv(f'~/resampling/data/{prefix}_site_diversity_{i}.csv', index=False)
        print(f'wrote results file: ~/resampling/data/{prefix}_site_diversity_{i}.csv')
        hetero_df.to_csv(f'~/resampling/data/{prefix}_heterozygosity_{i}.csv', index=False)
        print(f'wrote results file: ~/resampling/data/{prefix}_heterozygosity_{i}.csv')
        # uncomment this if you want to run for all paris of diploid_size and seq_len
        # break
