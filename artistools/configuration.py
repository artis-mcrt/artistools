import subprocess
import psutil
from pathlib import Path

# num_processes = 1

# count the cores (excluding the efficiency cores on ARM)
try:
    num_processes = int(subprocess.run(
        ['sysctl', '-n', 'hw.perflevel0.logicalcpu'], capture_output=True, text=True, check=True).stdout)
except subprocess.CalledProcessError:
    try:
        num_processes = int(subprocess.run(
            ['sysctl', '-n', 'hw.logicalcpu'], capture_output=True, text=True, check=True).stdout)
    except subprocess.CalledProcessError:
        num_processes = max(1, int(psutil.cpu_count(logical=False)) - 2)

# print(f'Using {num_processes} processes')

config = {}
config['enable_diskcache'] = True
config['num_processes'] = num_processes
config['figwidth'] = 5
config['codecomparisondata1path'] = Path(
    '/Users/luke/Library/Mobile Documents/com~apple~CloudDocs/GitHub/sn-rad-trans/data1')

config['codecomparisonmodelartismodelpath'] = Path('/Volumes/GoogleDrive/My Drive/artis_runs/weizmann/')

config['path_artistools_repository'] = Path(__file__).absolute().parent.parent
config['path_artistools_dir'] = Path(__file__).absolute().parent  # the package path
config['path_datadir'] = Path(__file__).absolute().parent / 'data'
config['path_testartismodel'] = Path(config['path_artistools_repository'], 'tests', 'data', 'testmodel')
config['path_testoutput'] = Path(config['path_artistools_repository'], 'tests', 'output')
