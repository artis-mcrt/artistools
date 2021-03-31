import multiprocessing
from pathlib import Path

# num_processes = 1
# num_processes = multiprocessing.cpu_count() - 1
num_processes = max(1, int(multiprocessing.cpu_count() / 2))
# print(f'Using {num_processes} processes')

enable_diskcache = True

figwidth = 5

config = {}
config['codecomparisondata1path'] = Path('/Volumes/GoogleDrive/My Drive/GitHub/sn-rad-trans/data1')
config['codecomparisonmodelartismodelpath'] = Path('/Volumes/GoogleDrive/My Drive/artis_runs/weizmann/')

config['path_artistools_repository'] = Path(__file__).absolute().parent.parent
config['path_artistools_sourcedir'] = Path(__file__).absolute().parent
config['path_datadir'] = Path(__file__).absolute().parent / 'data'
config['path_testartismodel'] = Path(config['path_artistools_repository'], 'tests', 'data', 'testmodel')
