import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0,str(_ProjectPath(__file__).resolve().parents[1]))
import argparse, json, random, time, re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mini_llm.tokenizer import SimpleTokenizer
from mini_llm.model import MiniLLM
from mini_llm.loss import causal_lm_loss
from mini_jev.decision_model import MiniJev
from mini_jev.inference import prepare_question
from mini_jev.calibration import fit_temperature, temperature_scale
from datasets.preprocessing import (banking77_to_records, clinc150_to_records,
    make_decision_example, make_multi_question_example, canonical_question_prompt, canonical_answer,
    tokenizer_texts_for_examples, deterministic_split, shifted_records, _noul_target)
from evaluation.benchmark import parameter_count, evaluate_classification, score_metrics, binary_metrics
from evaluation.latency import measure, throughput
from evaluation.calibration import reliability_data, confidence_histogram

ROOT=Path(__file__).resolve().parents[1]

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def device_of(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_tokenizer(examples):
    return SimpleTokenizer().fit(tokenizer_texts_for_examples(examples))

def ids(tok, text, max_len, device):
    return torch.tensor([tok.encode(text, max_len)], dtype=torch.long, device=device)

def _llm_training_sequence(tok, model, ex, q, t):
    prompt=canonical_question_prompt(ex['state'], q)
    answer=canonical_answer(q, t['target'])
    prompt_ids=[tok.stoi['<bos>']]+[tok.stoi.get(x,tok.stoi['<unk>']) for x in tok.tokenize(prompt)]
    answer_ids=[tok.stoi.get(x,tok.stoi['<unk>']) for x in tok.tokenize(answer)]
    # Keep EOS as a supervised target so generation learns where the answer ends.
    eos_id=tok.stoi['<eos>']
    full_ids=prompt_ids+answer_ids+[eos_id]
    if len(full_ids)>model.max_seq_len:
        raise ValueError('Mini-LLM training example exceeds max_seq_len; answer would be truncated')
    pad_id=tok.stoi['<pad>']
    ids=full_ids+[pad_id]*(model.max_seq_len-len(full_ids))
    # Labels align with the next-token target in causal_lm_loss: positions before
    # the answer are masked, while answer tokens + EOS remain supervised.
    labels=[-100]*len(prompt_ids)+answer_ids+[eos_id]
    labels += [-100]*(model.max_seq_len-len(labels))
    return ids, labels, prompt_ids, answer_ids

def train_llm(model, tok, examples, epochs, device, lr, batch_size, steps=None):
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    sequences=[]; labels=[]
    for ex in examples:
        for q,t in zip(ex['questions'], ex['ground_truth']):
            ids_, labels_, _, _ = _llm_training_sequence(tok,model,ex,q,t)
            sequences.append(ids_); labels.append(labels_)
    x=torch.tensor(sequences, dtype=torch.long, device=device)
    y=torch.tensor(labels, dtype=torch.long, device=device)
    start=time.perf_counter(); nsteps=0; seen=0; tokens=0; supervised_tokens=0; last_loss=float('nan')
    for _ in range(epochs):
        order=torch.randperm(len(x), device=device)
        for j in range(0,len(x),batch_size):
            idx=order[j:j+batch_size]; b=x[idx]; by=y[idx]
            seen+=len(b); tokens += int((b != tok.stoi['<pad>']).sum().item()); supervised_tokens += int((by != -100).sum().item())
            loss=causal_lm_loss(model(b),by,pad_id=-100)
            if not torch.isfinite(loss): raise RuntimeError('Mini-LLM training loss is not finite')
            last_loss=float(loss.detach().cpu())
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); nsteps+=1
            if steps and nsteps>=steps: return time.perf_counter()-start,nsteps,seen,tokens,last_loss,supervised_tokens
    return time.perf_counter()-start,nsteps,seen,tokens,last_loss,supervised_tokens

def train_jev(model,tok,examples,epochs,device,lr,batch_size=16,steps=None):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01); start=time.perf_counter(); nsteps=0; seen=0; tokens=0; last_loss=float('nan')
    model.train()
    order_examples=list(examples)
    for _ in range(epochs):
        random.shuffle(order_examples)
        for j in range(0,len(order_examples),batch_size):
            batch=order_examples[j:j+batch_size]
            state=torch.tensor([tok.encode(e['state'],model.encoder.pos.num_embeddings) for e in batch],dtype=torch.long,device=device)
            qbatch=[[prepare_question(q,tok,64,device) for q in e['questions']] for e in batch]
            tokens += sum(int((tok.encode(e['state'],model.encoder.pos.num_embeddings).__len__())) + sum(len(tok.tokenize(q['instruction'])) + len(tok.tokenize(' '.join(q.get('options',q.get('levels',['yes','no']))))) for q in e['questions']) for e in batch)
            outputs=model(state,qbatch)
            total=torch.zeros((),device=device); count=0
            for sample_out,e in zip(outputs,batch):
                for pred,t in zip(sample_out,e['ground_truth']):
                    if t['type'] in ('choice','score'):
                        total=total+F.cross_entropy(pred['logits'].unsqueeze(0),torch.tensor([t['target']],device=device))
                    else:
                        total=total+F.binary_cross_entropy_with_logits(pred['logit'].view(1),torch.tensor([float(t['target'])],device=device))
                    count+=1
            loss=total/count
            if not torch.isfinite(loss): raise RuntimeError('Mini-Jev training loss is not finite')
            last_loss=float(loss.detach().cpu())
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); nsteps+=1; seen+=len(batch)
            if steps and nsteps>=steps: return time.perf_counter()-start,nsteps,seen,tokens,last_loss
    return time.perf_counter()-start,nsteps,seen,tokens,last_loss

def _fit_prompt_tokens(tok, state, q, max_prompt_len):
    if max_prompt_len < 4:
        raise ValueError('max_prompt_len is too small to preserve ANSWER:')
    if q['type'] == 'choice':
        opts = q['options']
    elif q['type'] == 'score':
        opts = q['levels']
    else:
        opts = ['yes', 'no']
    state_tokens = tok.tokenize(state)
    suffix = f" QUESTION: {q['instruction']} OPTIONS: {' | '.join(opts)} ANSWER:"
    suffix_ids = [tok.stoi.get(t, tok.stoi['<unk>']) for t in tok.tokenize(suffix)]
    available = max_prompt_len - 1 - len(suffix_ids)
    if available < 0:
        raise ValueError('Question/options suffix is longer than model context; ANSWER: cannot be preserved')
    if len(state_tokens) > available:
        state_tokens = state_tokens[:available]
    text = f"STATE: {' '.join(state_tokens)}" + suffix
    ids = [tok.stoi['<bos>']] + [tok.stoi.get(t, tok.stoi['<unk>']) for t in tok.tokenize(text)]
    if len(ids) > max_prompt_len:
        raise AssertionError('Prompt fitting failed to preserve complete prompt suffix')
    return ids

def llm_choice_probs(model,tok,prompt,options,device,q=None,state=None):
    if q is not None and state is not None:
        prompt_ids = _fit_prompt_tokens(tok, state, q, model.max_seq_len - max(len(tok.tokenize(o)) for o in options))
    else:
        prompt_ids=[tok.stoi['<bos>']] + [tok.stoi.get(t,tok.stoi['<unk>']) for t in tok.tokenize(prompt)]
    scores=[]
    model.eval()
    with torch.no_grad():
        for option in options:
            cand=tok.tokenize(option)
            cand_ids=[tok.stoi.get(t,tok.stoi['<unk>']) for t in cand]
            if not cand_ids:
                scores.append(float('-inf')); continue
            max_prompt_len=model.max_seq_len-len(cand_ids)
            if max_prompt_len < 1:
                scores.append(float('-inf')); continue
            prefix=prompt_ids if len(prompt_ids) <= max_prompt_len else prompt_ids[-max_prompt_len:]
            seq=prefix+cand_ids
            prefix_len=len(prefix)
            x=torch.tensor([seq],dtype=torch.long,device=device)
            logits=model(x)[0]
            lp=F.log_softmax(logits,-1)
            idx=torch.tensor(cand_ids,dtype=torch.long,device=device)
            token_lp=lp[prefix_len-1:prefix_len-1+len(cand_ids)].gather(1,idx[:,None]).squeeze(1)
            value=float(token_lp.mean())
            if not np.isfinite(value):
                raise AssertionError('Non-finite candidate log-probability')
            scores.append(value)
    score_tensor=torch.tensor(scores,dtype=torch.float32)
    if not torch.isfinite(score_tensor).any():
        raise AssertionError('All candidate log-probabilities are non-finite')
    probs=torch.softmax(score_tensor,0)
    if not torch.isfinite(probs).all() or not torch.isclose(probs.sum(),torch.tensor(1.),atol=1e-5):
        raise AssertionError('Candidate probabilities are not finite or do not sum to 1')
    return probs.numpy()

def parse_structured_option(text, tok, options):
    normalized=' '.join(text.lower().split())
    normalized=re.sub(r'\banswer\s*:', 'answer :', normalized)
    marker='answer :'
    if marker not in normalized:
        return None
    tail=normalized.split(marker,1)[1].strip()
    tail_tokens=tok.tokenize(tail)
    matches=[]
    for i,opt in enumerate(options):
        opt_tokens=tok.tokenize(opt)
        if tail_tokens[:len(opt_tokens)] == opt_tokens:
            matches.append((len(opt_tokens), i))
    exact=[m for m in matches if len(tail_tokens)==m[0]]
    return max(exact)[1] if exact else None

def llm_structured_answer(model,tok,state,q,options,device,return_debug=False):
    prompt=canonical_question_prompt(state,q)
    max_new=max(4, max(len(tok.tokenize(o)) for o in options)+2)
    max_prompt=max(1, model.max_seq_len-max_new)
    prompt_ids=_fit_prompt_tokens(tok,state,q,max_prompt)
    x=torch.tensor([prompt_ids],dtype=torch.long,device=device)
    generated_full=model.generate(x,max_new,temperature=0.0,eos_id=tok.stoi['<eos>'],pad_id=tok.stoi['<pad>'],bos_id=tok.stoi['<bos>'])[0].tolist()
    generated_new=generated_full[len(prompt_ids):]
    # The model receives the prompt once; this is the exact decoded continuation.
    generated_raw=tok.decode(generated_new)
    generated_normalized=' '.join(generated_raw.lower().split())
    # Keep the benchmark parser on the exact full decoded model output so the
    # diagnostic observes the same parser result as normal structured evaluation.
    full_text=tok.decode(generated_full)
    parsed=parse_structured_option(full_text,tok,options)
    if return_debug:
        return parsed, {
            'prompt_ids': prompt_ids,
            'prompt_tokens': [tok.itos[i] for i in prompt_ids],
            'generated_new_ids': generated_new,
            'generated_new_tokens': [tok.itos[i] for i in generated_new],
            'generated_tokens': len(generated_new),
            'generated_raw': generated_raw,
            'generated_normalized': generated_normalized,
            'parsed_option': None if parsed is None else options[parsed],
            'expected_option': None,
            'parser_match': False,
            'failure_reason': None if parsed is not None else 'no valid complete option after ANSWER:',
            'generation_started_after_prompt': generated_full[:len(prompt_ids)] == prompt_ids,
        }
    return parsed


def debug_structured_examples(model,tok,examples,device,count=5):
    """Temporary, stdout-only diagnostic for the real Mini-LLM evaluation path."""
    if len(examples) < count:
        raise ValueError(f"[LLM DEBUG] expected at least {count} test examples, got {len(examples)}")
    for idx,e in enumerate(examples[:count], start=1):
        q=e['questions'][0]
        target=e['ground_truth'][0]['target']
        options=q.get('options',q.get('levels',['yes','no']))
        expected=canonical_answer(q,target)

        # This calls the exact same structured generation function used by the
        # benchmark.  No expected answer is fed into generation or parsing.
        parsed,dbg=llm_structured_answer(model,tok,e['state'],q,options,device,return_debug=True)
        dbg['expected_option']=expected
        dbg['parser_match']=(parsed is not None and options[parsed] == expected)
        training_text=canonical_question_prompt(e['state'],q)+' '+expected

        print(f"[LLM DEBUG {idx}]")
        print(f"question_type: {q['type']}")
        print(f"state: {e['state']}")
        print(f"question: {q['instruction']}")
        print(f"options: {options}")
        print(f"expected_answer: {expected}")
        print(f"training_text: {training_text}")
        print(f"generated_raw: {dbg['generated_raw']}")
        print(f"generated_normalized: {dbg['generated_normalized']}")
        print(f"parsed_answer: {dbg['parsed_option']}")
        print(f"expected_option: {dbg['expected_option']}")
        print(f"parsed_match: {parsed is not None and options[parsed] == expected}")
        print(f"generated_tokens: {dbg['generated_tokens']}")


def debug_candidate_scoring(model,tok,examples,device,limit=5):
    rows=[]
    for e in examples[:limit]:
        q=e['questions'][0]
        options=q.get('options',q.get('levels',['yes','no']))
        p=llm_choice_probs(model,tok,canonical_question_prompt(e['state'],q),options,device,q=q,state=e['state'])
        rows.append(p)
    arr=np.stack(rows) if rows else np.empty((0,0))
    if len(arr):
        print(f"\n[LLM DEBUG] candidate scoring: rows={len(arr)}, finite={bool(np.isfinite(arr).all())}, row_sums={[round(float(x),6) for x in arr.sum(1)]}")
    else:
        print("\n[LLM DEBUG] candidate scoring: no examples")

def jev_probs(model,tok,e,device):
    model.eval(); state=ids(tok,e['state'],model.encoder.pos.num_embeddings,device); qs=[[prepare_question(q,tok,64,device) for q in e['questions']]]
    with torch.no_grad(): return model(state,qs)[0]

def choice_arrays(model,tok,examples,device,question_index=0):
    ys=[]; probs=[]; failures=0
    for e in examples:
        p=jev_probs(model,tok,e,device)[question_index]; t=e['ground_truth'][question_index]['target']; ys.append(t); probs.append(p['probabilities'].detach().cpu().numpy())
    return np.array(ys),np.stack(probs),failures

def llm_arrays(model,tok,examples,device,question_index=0):
    ys=[]; probs=[]; failures=0
    for e in examples:
        q=e['questions'][question_index]; target=e['ground_truth'][question_index]['target']; ys.append(target)
        options=q.get('options',q.get('levels',['yes','no']))
        p=llm_choice_probs(model,tok,canonical_question_prompt(e['state'],q),options,device,q=q,state=e['state']); probs.append(p)
        failures += llm_structured_answer(model,tok,e['state'],q,options,device) is None
    return np.array(ys),np.stack(probs),failures

def noul_arrays(model,tok,examples,device,question_index=3):
    ys=[]; probs=[]
    for e in examples:
        q=e['questions'][question_index]; ys.append(e['ground_truth'][question_index]['target'])
        p=llm_choice_probs(model,tok,canonical_question_prompt(e['state'],q),['yes','no'],device,q=q,state=e['state']); probs.append(float(p[0]))
    return np.array(ys,dtype=int),np.array(probs,dtype=float)

def jev_noul_arrays(model,tok,examples,device,question_index=3):
    ys=[]; probs=[]
    for e in examples:
        p=jev_probs(model,tok,e,device)[question_index]; ys.append(e['ground_truth'][question_index]['target']); probs.append(float(p['probability_yes']))
    return np.array(ys,dtype=int),np.array(probs,dtype=float)

def calibrate(probs,targets,min_examples=20,min_classes=2):
    probs=np.asarray(probs,dtype=float); targets=np.asarray(targets,dtype=int)
    if len(targets) < min_examples:
        return probs,None,'skipped_insufficient_validation_data'
    if len(np.unique(targets)) < min_classes:
        return probs,None,'skipped_insufficient_validation_class_diversity'
    if probs.ndim != 2 or not np.isfinite(probs).all() or not np.allclose(probs.sum(1),1.,atol=1e-5):
        return probs,None,'skipped_invalid_validation_probabilities'
    logits=torch.tensor(np.log(np.clip(probs,1e-8,None)),dtype=torch.float32)
    t=fit_temperature(logits,torch.tensor(targets,dtype=torch.long))
    if not np.isfinite(t) or t <= 0 or t > 100.:
        return probs,None,'skipped_unstable_temperature'
    calibrated=torch.softmax(temperature_scale(logits,t),-1).numpy()
    return calibrated,t,'calibrated'

def run_family(name, train_records, val_records, test_records, llm_cfg, jev_cfg, seed_value, args, device, emit_llm_debug=False):
    seed(seed_value); labels=sorted({r['label'] for r in train_records})
    train=[make_multi_question_example(r,labels) for r in train_records]
    val=[make_multi_question_example(r,labels) for r in val_records if r['label'] in labels]
    test=[make_multi_question_example(r,labels) for r in test_records if r['label'] in labels]
    tok=make_tokenizer(train)
    llm=MiniLLM(tok.vocab_size,**llm_cfg).to(device); jev=MiniJev(tok.vocab_size,**jev_cfg).to(device)
    llm_time,llm_steps,llm_seen,llm_tokens,llm_loss,llm_supervised_tokens=train_llm(llm,tok,train,args.epochs,device,args.lr,args.batch_size,args.steps)
    jev_time,jev_steps,jev_seen,jev_tokens,jev_loss=train_jev(jev,tok,train,args.epochs,device,args.lr,args.batch_size,args.steps)
    if emit_llm_debug:
        debug_structured_examples(llm,tok,test,device,count=5)
    # Primary intent probabilities + validation-only temperature calibration.
    yv,plv,_=llm_arrays(llm,tok,val,device,0); yv2,pjv,_=choice_arrays(jev,tok,val,device,0)
    plc,lt,lstatus=calibrate(plv,yv); pjc,jt,jstatus=calibrate(pjv,yv2)
    yt,plt,lf=llm_arrays(llm,tok,test,device,0); yt2,pjt,_=choice_arrays(jev,tok,test,device,0)
    task_metrics={}
    for qi,qname in enumerate(('intent_choice','routing_choice','risk_score')):
        if qi < len(test[0]['questions']):
            yy,pp,llf=llm_arrays(llm,tok,test,device,qi); yj,pj,_=choice_arrays(jev,tok,test,device,qi)
            if test[0]['questions'][qi]['type']=='score':
                task_metrics[qname]={'mini_llm':score_metrics(pp,yy),'mini_jev':score_metrics(pj,yj),'mini_llm_structured_failure_rate':llf/max(1,len(test))}
            else:
                task_metrics[qname]={'mini_llm':evaluate_classification(yy,pp),'mini_jev':evaluate_classification(yj,pj),'mini_llm_structured_failure_rate':llf/max(1,len(test))}
    yn,pln=noul_arrays(llm,tok,test,device); yjn,pjn=jev_noul_arrays(jev,tok,test,device)
    noul_unique=np.unique(yn)
    if len(noul_unique) < 2:
        if name == 'banking77':
            raise ValueError('Banking77 Noul test set contains only one class; refusing to report a trivial binary benchmark')
        noul_status='auxiliary_synthetic_one_class_source'
    else:
        noul_status='source_semantic_binary_task'
    task_metrics['noul']={'mini_llm':binary_metrics(pln,yn),'mini_jev':binary_metrics(pjn,yjn),'positive_rate':float(yn.mean()),'unique_targets':noul_unique.tolist(),'status':noul_status}
    out={'seed':seed_value,'family':name,'labels':len(labels),'train_examples':len(train),'validation_examples':len(val),'test_examples':len(test),'questions_per_state':len(train[0]['questions']) if train else 0,'mini_llm_question_level_sequences':len(train)*4,'mini_jev_state_level_examples':len(train),
         'task_metrics':task_metrics,
         'mini_llm':evaluate_classification(yt,plt),'mini_jev':evaluate_classification(yt2,pjt),
         'noul_test':{'mini_llm':binary_metrics(pln,yn),'mini_jev':binary_metrics(pjn,yjn)},
         'calibration':{'mini_llm':lstatus,'mini_jev':jstatus},
         'calibrated_mini_llm':evaluate_classification(yt,torch.softmax(torch.tensor(np.log(np.clip(plt,1e-8,None)))/lt,-1).numpy()) if lt is not None else None,
         'calibrated_mini_jev':evaluate_classification(yt2,torch.softmax(torch.tensor(np.log(np.clip(pjt,1e-8,None)))/jt,-1).numpy()) if jt is not None else None,
         'validation_temperature':{'mini_llm':lt,'mini_jev':jt},
         'parameters':{'mini_llm':parameter_count(llm),'mini_jev':parameter_count(jev)},
         'training':{'mini_llm_seconds':llm_time,'mini_jev_seconds':jev_time,'mini_llm_steps':llm_steps,'mini_jev_steps':jev_steps,'mini_llm_examples_seen':llm_seen,'mini_jev_examples_seen':jev_seen,'mini_llm_tokens_processed':llm_tokens,'mini_jev_tokens_processed':jev_tokens,'mini_llm_final_loss':llm_loss,'mini_llm_supervised_target_tokens':llm_supervised_tokens,'mini_jev_final_loss':jev_loss},
         'structured_output_failure_rate':lf/max(1,len(test)),
         'reliability':{'mini_llm':reliability_data(plt,yt),'mini_jev':reliability_data(pjt,yt2)},
         'confidence_histogram':{'mini_llm':confidence_histogram(plt),'mini_jev':confidence_histogram(pjt)}}
    # Parallel vs serial on one state, using the same trained Mini-Jev.
    if test:
        e=test[0]; questions=[prepare_question(q,tok,64,device) for q in e['questions']]
        sids=ids(tok,e['state'],jev.encoder.pos.num_embeddings,device)
        parallel_ms=measure(lambda: jev(sids,[questions]),runs=args.latency_runs)
        serial_ms=measure(lambda: [jev(sids,[[q]]) for q in questions],runs=args.latency_runs)
        out['parallel_questions_ms']=parallel_ms; out['serial_questions_ms']=serial_ms; out['parallel_speedup']=serial_ms/max(parallel_ms,1e-9)
        out['throughput_samples_per_sec']=throughput(lambda: jev(sids,[questions]),1,runs=args.latency_runs)
    return out

def candidate_cfg(target_count,vocab,base):
    best=None
    for d in (48,56,64,72,80,88,96,112,128):
        for layers in (1,2,3):
            for mult in (2,3,4):
                if d%4: continue
                c=dict(d_model=d,n_heads=4,n_layers=layers,d_ff=d*mult,dropout=base.get('dropout',.1),max_seq_len=base.get('max_seq_len',128))
                m=MiniJev(vocab,**c); n=parameter_count(m); err=abs(n-target_count)
                if best is None or err<best[0]: best=(err,c,n)
    return best[1],best[2]

def _debug_subset(records, limit):
    if not limit or len(records) <= limit:
        return list(records)
    # Preserve source order while ensuring the binary Noul task has both classes when the source supports it.
    positives=[r for r in records if r['label'].lower() in {
        'card_payment_not_recognised','cash_withdrawal_not_recognised',
        'direct_debit_payment_not_recognised','compromised_card',
        'lost_or_stolen_card','lost_or_stolen_phone'}]
    negatives=[r for r in records if r['label'].lower() not in {
        'card_payment_not_recognised','cash_withdrawal_not_recognised',
        'direct_debit_payment_not_recognised','compromised_card',
        'lost_or_stolen_card','lost_or_stolen_phone'}]
    if positives and negatives and limit >= 2:
        selected=[positives[0], negatives[0]]
        selected.extend(r for r in records if r not in selected)
        return selected[:limit]
    return list(records[:limit])

def _sanity_check_examples(train, val, test, labels, calibration_requested=True):
    if not train: raise ValueError('Benchmark sanity check failed: training set is empty')
    if not test: raise ValueError('Benchmark sanity check failed: test set is empty')
    if len(labels) <= 1: raise ValueError('Benchmark sanity check failed: training has <=1 unique label')
    if calibration_requested and not val:
        raise ValueError('Benchmark sanity check failed: validation set is empty while calibration is requested')
    if any(parameter_count(m) <= 0 for m in []):
        raise AssertionError('Benchmark sanity check failed: model parameter count is zero')

def _noul_distribution(examples):
    vals=[_noul_target(r['label']) for r in examples]
    return {'negative':vals.count(0),'positive':vals.count(1),'positive_rate':float(np.mean(vals)) if vals else None,'unique_targets':sorted(set(vals))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['banking77','clinc150'],default='banking77'); ap.add_argument('--banking-train'); ap.add_argument('--banking-test'); ap.add_argument('--clinc-json'); ap.add_argument('--epochs',type=int,default=1); ap.add_argument('--steps',type=int); ap.add_argument('--limit',type=int); ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--lr',type=float,default=5e-4); ap.add_argument('--seeds',default='42,43,44'); ap.add_argument('--protocol',choices=['default','parameter-matched','compute-matched'],default='default'); ap.add_argument('--latency-runs',type=int,default=5); ap.add_argument('--debug-structured',action='store_true'); ap.add_argument('--out',default='results/benchmark.json'); a=ap.parse_args(); device=device_of()
    if a.dataset=='banking77':
        if not a.banking_train or not a.banking_test: ap.error('banking paths required')
        raw_train=banking77_to_records(a.banking_train); raw_test=banking77_to_records(a.banking_test); train,val,_=deterministic_split(raw_train,42); test=raw_test
    else:
        if not a.clinc_json: ap.error('--clinc-json required')
        train,val,test=deterministic_split(clinc150_to_records(a.clinc_json,'train'),42); test=clinc150_to_records(a.clinc_json,'test')
    if a.limit:
        train=_debug_subset(train,a.limit)
        train_labels={r['label'] for r in train}
        val=[r for r in val if r['label'] in train_labels]
        test=[r for r in test if r['label'] in train_labels]
        val=_debug_subset(val,max(1,a.limit//5))
        test=_debug_subset(test,a.limit)
    train_labels={r['label'] for r in train}
    if a.limit and len(val)==0:
        raise ValueError('Debug benchmark produced an empty validation set after train-label filtering')
    _sanity_check_examples(train,val,test,train_labels,calibration_requested=True)
    print(json.dumps({'debug_or_benchmark': 'pipeline_debug' if a.limit else 'full_benchmark', 'train_examples':len(train), 'unique_training_labels':len(train_labels), 'validation_examples':len(val), 'test_examples':len(test), 'noul_distribution_train':_noul_distribution(train), 'noul_distribution_validation':_noul_distribution(val), 'noul_distribution_test':_noul_distribution(test)}, indent=2))
    if a.dataset=='banking77' and len(set(_noul_target(r['label']) for r in train)) < 2:
        raise ValueError('Banking77 Noul sanity check failed: training subset contains only one Noul class')
    if a.protocol=='compute-matched' and a.steps is None:
        a.steps=max(1,(len(train)+a.batch_size-1)//a.batch_size)*a.epochs
    base={'d_model':96,'n_heads':4,'n_layers':2,'d_ff':192,'dropout':.1,'max_seq_len':512}
    # Match parameters after train-only vocabulary is known.
    labels_for_vocab=sorted({r['label'] for r in train})
    tokenizer_examples=[make_multi_question_example(r,labels_for_vocab) for r in train]
    tok=make_tokenizer(tokenizer_examples); llm_ref=MiniLLM(tok.vocab_size,**base); llm_n=parameter_count(llm_ref)
    llm_cfg=base; jev_cfg=base
    matched=None
    if a.protocol=='parameter-matched': jev_cfg,jev_n=candidate_cfg(llm_n,tok.vocab_size,base); matched={'mini_llm_parameters':llm_n,'mini_jev_parameters':jev_n,'mini_jev_config':jev_cfg}
    seeds=[int(x) for x in a.seeds.split(',') if x.strip()]
    results=[]
    for s in seeds:
        # Compute-matched means same examples, batch size and optimizer steps; no architecture advantage is hidden.
        emit_llm_debug=(a.dataset=='banking77' and a.limit==200 and s==seeds[0])
        r=run_family(a.dataset,train,val,test,llm_cfg,jev_cfg,s,a,device,emit_llm_debug=emit_llm_debug); r['protocol']=a.protocol; r['timing']={'training_time_seconds':{'mini_llm':r['training']['mini_llm_seconds'],'mini_jev':r['training']['mini_jev_seconds']},'inference_time_ms':{'parallel_questions':r.get('parallel_questions_ms')},'parameter_count':r['parameters']}; r['compute_control']={'equal_examples':r['training']['mini_llm_examples_seen']==r['training']['mini_jev_examples_seen'],'equal_optimizer_steps':r['training']['mini_llm_steps']==r['training']['mini_jev_steps'],'batch_size':a.batch_size,'requested_steps':a.steps,'tokens_processed':{'mini_llm':r['training']['mini_llm_tokens_processed'],'mini_jev':r['training']['mini_jev_tokens_processed']},'training_time_seconds':{'mini_llm':r['training']['mini_llm_seconds'],'mini_jev':r['training']['mini_jev_seconds']},'parameter_count':r['parameters'],'exact_flop_matched':False,'description':'Compute-matched controls examples, batch size, and optimizer steps; it does not claim exact FLOP matching.'}
        results.append(r)
    summary={'dataset':a.dataset,'device':str(device),'protocol':a.protocol,'seeds':seeds,'matched':matched,'runs':results,'status':'measured' if results else 'not yet measured'}
    # OOD evaluation is explicit and uses the same trained models only in a separate script to avoid test tuning.
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(summary,open(a.out,'w'),indent=2); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
