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
import simulation as sim
import observation as obs
from datetime import datetime
from div import columns, location, calculate_ints


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
    # count alleles within each population at the selected sites and ind
    ac1 = ga[sites_index][:, popA_samples, :].count_alleles()
    ac2 = ga[sites_index][:, popB_samples, :].count_alleles()
    # calculate Hudson's Fst (weighted)
    num, den = allel.hudson_fst(ac1, ac2)
    fst = np.sum(num) / np.sum(den)
    return fst


def experiment(num_exp, num_obs, confidence=0.95, diploid_size=200,
               split_time=50, seq_len=1e9, rec_rate=1e-8, mut_rate=1e-8, seed=None):
    """Run experiment for resampling of site diversity
    @num_exp: number of experiment to run
    @num_obs: number of observations for each num_ind and max_sites
    @confidence: confidence level
    @diploid_size = the population size of each population,
        also the size of the ancestral population
    @split_time = current populations split from the
        ancestral population this many generations ago
    @seq_len = length of the genome, units ~ base-pairs
    @rec_rate = recombination rate, units = rate per bp, per generation
    @mut_rate = mutation rate, units = rate per bp, per generation
    @seed: set seed for the experiment
    """
    result = []
    n_block = int(seq_len // 5e6)
    np.random.seed(seed)
    # generate seeds for population ts
    ts_seeds = np.random.randint(0, 2 ** 32 - 1, num_exp)

    for exp in range(num_exp):
        start = datetime.now()
        print(f'\n Experiment {exp} starts at:', exp)

        pop_ts = sim.sim_population(
            diploid_size=diploid_size,
            split_time=split_time,
            seq_len=seq_len,
            rec_rate=rec_rate,
            mut_rate=mut_rate,
            seed=ts_seeds[exp]
        )

        pop_ts_fst = get_fst(pop_ts)
        print('Population num sites is:', pop_ts.num_sites)
        print('Population site fst:', pop_ts_fst)

        num_ind_list = [50]
        max_sites_list = [5000, 20000, 50000]
        assert max(num_ind_list) <= pop_ts.num_individuals and max(max_sites_list) <= pop_ts.num_sites, \
            "Number of ind and max_sites must be smaller than the population"

        # generate seeds for all possible observation ts
        np.random.seed(ts_seeds[exp])
        obs_seeds = np.random.randint(0, 2 ** 32 - 1, size=(len(num_ind_list) * len(max_sites_list), num_obs))

        row = 0
        for num_ind in num_ind_list:
            for max_sites in max_sites_list:
                print(f'The shape of observed population (num_ind, max_sites) is: {num_ind, max_sites}')
                position = np.zeros((num_obs, 15))

                for j in range(num_obs):
                    obs_ts = obs.Fst(pop_ts, num_ind, max_sites, seed=obs_seeds[row][j])

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

                result.append([exp, num_ind, max_sites, num_obs,
                               seq_len, rec_rate, mut_rate, pop_ts_fst]
                              + list(position.sum(0)))

        print(f'Experiment {exp} runs:', datetime.now() - start)
        print('Fst outcomes: \n')
        print(result[exp][8:])

    # save the results to csv file
    result_df = pd.DataFrame(result)
    result_df.columns = columns
    return result_df


if __name__ == '__main__':
    prefix = datetime.now().strftime("%m%d%H%M")
    seed = 1
    diploid_size = [200, 1000, 1500]
    seq_len = [1e8, 5e8, 1e9]
    for i, (d, s) in enumerate(zip(diploid_size, seq_len)):
        df = experiment(num_exp=1, num_obs=100, diploid_size=d, seq_len=s, confidence=0.95, seed=seed)
        df.to_csv(f'../data/{prefix}_fst.csv_{i}', index=False)
        # uncomment this if you want to run for all paris of diploid_size and seq_len
        break
