import numpy as np
from numpy.random import normal
from pandas import DataFrame

NUM_DATASET = 1000
DATA_LEN = 1000

def main():
  data_x = np.linspace(0, 10, DATA_LEN)
  l = [np.sin(2.0 * np.pi * data_x) + normal(loc=0.0, scale=0.3, size=DATA_LEN) for _ in range(NUM_DATASET)]
  df = DataFrame(np.array(l))
  train = df[0:int(NUM_DATASET*0.8)]
  test = df[int(NUM_DATASET*0.8):NUM_DATASET]
  train.T.to_csv('data/train_data.csv', index=False, header=False)
  test.T.to_csv('data/test_data.csv', index=False, header=False)

if __name__ == '__main__':
  main()