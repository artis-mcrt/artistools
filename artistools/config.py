import psutil
from pathlib import Path

# num_processes = 1
num_processes = max(1, int(psutil.cpu_count(logical=False)) - 2)
# print(f'Using {num_processes} processes')

enable_diskcache = True

figwidth = 5

config = {}
config['codecomparisondata1path'] = Path(
    '/Users/luke/Library/Mobile Documents/com~apple~CloudDocs/GitHub/sn-rad-trans/data1')

config['codecomparisonmodelartismodelpath'] = Path('/Volumes/GoogleDrive/My Drive/artis_runs/weizmann/')

config['path_artistools_repository'] = Path(__file__).absolute().parent.parent
config['path_artistools_sourcedir'] = Path(__file__).absolute().parent
config['path_datadir'] = Path(__file__).absolute().parent / 'data'
config['path_testartismodel'] = Path(config['path_artistools_repository'], 'tests', 'data', 'testmodel')
config['path_testoutput'] = Path(config['path_artistools_repository'], 'tests', 'output')

config['gsimerger_trajroot'] = Path(
    '/Users/luke/Library/Mobile Documents/com~apple~CloudDocs/Archive/Astronomy/Mergers/SFHo/')
