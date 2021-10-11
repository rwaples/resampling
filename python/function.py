"""This file contains functions and global values that are used in both div.py and fst.py

Last Updated Date: Oct 11, 2021
"""
import numpy as np
import intervals as ci
import allel

columns = ['exp_seed', 'experiment', 'tree_sequence_seed',
           'seq_len', 'rec_rate', 'mut_rate',
           'pop_num_ind', 'pop_num_sites', 'population_value',
           'obs_num_ind', 'intended_obs_num_sites', 'actual_obs_num_sites', 'num_observations',
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
    intervals -- a list of intervals (lower, upper) from bt, jk_one, and jk_mj resampling methods
    pop_val -- float, the population value, for example (pop_ts_diversity)
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
    """Return the heterozygosity value of the given tree sequence
    ts -- tree sequence
    """
    geno = ts.genotype_matrix()
    ac = ((geno[:, ::2] + geno[:, 1::2]) == 1).sum(0)
    return np.mean(ac / ts.num_sites)


def calculate_ints(resample_values, confidence, obs_value):
    """calculate intervals for the resample_values
    resample_values -- bt, jk_one, jk_mj
    confidence -- confidence leve
    obs_value -- observed value
    """
    return [func_ints[i](resample_values[i], confidence, obs_value) for i in range(len(resample_values))]


def get_fst(ts):
    """returns Hudson's Fst
    ts -- tree sequence
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
