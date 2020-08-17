import pandas as pd
import numpy as np
csv_file = pd.read_csv("2015H_19_days_since_explosion.txt")
print(csv_file)

np.savetxt("2015H_19_days_since_explosion.txt", csv_file)

