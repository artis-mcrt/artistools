import multiprocessing

import artistools as at
import artistools.estimators.plotestimators

if __name__ == "__main__":
    multiprocessing.freeze_support()
    at.estimators.plotestimators.main()
