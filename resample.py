import numpy as np
from sklearn.model_selection import KFold


def site_diversity(ts):
    """return a numpy array of each site's diversity across the observed population
    
    """
    samples = np.arange(ts.num_samples)
    sites_index = np.arange(ts.num_sites)
    nsamples = len(samples)
    genos = ts.genotype_matrix()
    
    ac = genos[sites_index, :][:,samples].sum(1)
    
    npairs = int(nsamples * (nsamples - 1) / 2)  # total numbers of pairs of samples
    n_different_pairs = ac * (nsamples - ac) # number of those pairs that have different alleles
    return (n_different_pairs / npairs)
    

def bt_resample_sites(diversity, n_boot):
    """
    bootstrap resampling over sites 
    return a numpy array containing site_diversity of each bootstrap
    """
    
    num_sites = len(diversity)
    weights = np.random.multinomial(
        n=num_sites,
        pvals=np.ones(num_sites)/num_sites,
        size=n_boot
    )
    
    vals = np.mean((diversity * weights), axis=1)
    
    return vals
    
    
def jk_delete_one(diversity, obs_ts_diversity):
    '''
    delete one jackknife resample over sites
    Return a numpy array containing the psedovalues of site diversity
    '''
    
    num_sites = len(diversity)
    weigths = np.ones((num_sites, num_sites), dtype=int)
    np.fill_diagonal(weigths, 0)
    
    deleted = (diversity * weigths).sum(axis=1) / (num_sites - 1)

    vals = num_sites * obs_ts_diversity - (num_sites - 1) * deleted
    
    return deleted, vals
                         

def jk_delete_m(diversity, obs_ts_diversity, n_fold):
    '''
    delete_m jackknife resamle over sites (block jackknife with the same number of sites in each block)
    Return a numpy array containing the psedovalues of site diversity
    '''
    
    num_sites = len(diversity)
    weigths = np.ones((n_fold, num_sites), dtype=int)
    kf = KFold(n_splits=n_fold)
    
    # fill deleleted block with 0
    for i, (_, zero_index) in enumerate(kf.split(np.arange(num_sites))):
        weigths[i][zero_index] = 0
        
    deleted = (diversity * weigths).sum(axis=1) / ((weigths == 1).sum(axis=1))
    
    vals = n_fold * obs_ts_diversity - (n_fold - 1) * deleted
    
    return deleted, vals


def jk_delete_mj(diversity, obs_ts_diversity, n_fold):
    '''
    delete_mj jackknife resampling methods over sits 
    each block contains different number of sites
    return a numpy array containing the psedovalues of site diversity
    '''
    
    num_sites = len(diversity)
    
    # where to cut off the array
    random = np.random.randint(10, n_fold, n_fold - 1, dtype=int)
    cutoff = np.cumsum(random)
    
    # index of sites 
    index = np.arange(num_sites)
    index = np.split(index, cutoff)
    
    # number of sites in each fold
    sizes = [len(index[i]) for i in range(len(index))]

    weigths = np.ones((n_fold, num_sites), dtype=int)
    
    # fill deleleted block with 0
    for i, indices in enumerate(index):
        weigths[i][indices] = 0
    
    # site diversity aftering deleting one block
    deleted = (diversity * weigths).sum(axis=1) / ((weigths == 1).sum(axis=1))
    
    vals = n_fold * obs_ts_diversity - (n_fold - 1) * deleted
    
    return deleted, vals, np.array(sizes)