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