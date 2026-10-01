from pathlib import Path
from urllib.request import urlretrieve
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'data'; out.mkdir(exist_ok=True)
urls={
 'banking77_train.csv':'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv',
 'banking77_test.csv':'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv',
 'clinc150.json':'https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json'
}
for name,url in urls.items():
 p=out/name
 if not p.exists():
  try: print('downloading',name); urlretrieve(url,p)
  except Exception as e: print(f'Could not download {name}: {e}')
print('data:',out)
