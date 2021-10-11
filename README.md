# Resampling
### Simulations
1. Population size - 200, 1000, 5000
2. Genome size  - 1e8, 5e8,  1e9

### Observations
1. Number of observed individuals - 20, 50, 100
2. Number of observed sites 5000, 20000, 50000

# Run
```
# Change the working directory to the python file
cd python
```
```
# Run the file: replace the num_exp, num_obs, seed with
# the values you want. You can leave the seed empty if you
# don't want to have the seed for the experiment

# for div.py
python3 div.py num_exp num_obs seed

# for fst.py
python3 fst.py num_exp num_obs seed
```

# Data
Prefix of data file represents the (pop_ind)_(seq_len). \
The ending of data file represents the date (Month, Day) the data is generated. 

# Columns names
* exp_seed: the seed for the overall experiment;
* experiment: the experiment id;
* tree_sequence_seed: the seed for generating the population tree sequence;
* seq_len: sequence length;
* rec_rate: recreation rate;
* mut_rate: mutation rate;
* pop_num_ind: the number of individuals in one population. Note in fst, there are two populations;
* pop_num_sites: the overall number of sites;
* population_value: the statistical value of the population ts (diversity, hetero, or fst);
* obs_num_ind: number of observed individuals;
* intended_obs_num_sites: intended number of observed sites;
* actual_obs_num_sites: actual number of observed sites;
* num_observation: the number of observations in each experiment;
* _sites_lower: the number of times when population value is smaller than the lower bound of the confidence interval;
* _sites_within: the number of times when population value is within the confidence interval;
* _sites_above: the number of times when population value is larger than the upper bound of the confidence interval;
* bt: bootstrap resampling;
* jk_one: jackknife resampling (deleting one value at each time);
* jk_block: jackknife resampling (deleting a block values at each time);
