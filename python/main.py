import site_diversity
from datetime import datetime

if __name__ == '__main__':
    prefix = datetime.now().strftime("%m%d")
    site_diversity_df = site_diversity.experiment(num_exp=1, num_obs=1, confidence=0.95)
    # site_diversity_df.to_csv(f'../data/{prefix}_site_diversity.csv', index=False)
