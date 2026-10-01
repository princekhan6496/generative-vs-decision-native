from .preprocessing import banking77_to_records,clinc150_to_records,make_decision_example,deterministic_split
def load_banking77(train_csv,test_csv,seed=42):
 tr=banking77_to_records(train_csv); te=banking77_to_records(test_csv); labels=sorted({r['label'] for r in tr+te}); return tr,te,labels
def load_clinc150(json_path,seed=42):
 tr=clinc150_to_records(json_path,'train'); va=clinc150_to_records(json_path,'val'); te=clinc150_to_records(json_path,'test'); labels=sorted({r['label'] for r in tr if r['label']!='oos'}); return tr,va,te,labels
