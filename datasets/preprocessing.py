import json, random


def banking77_to_records(path):
    import csv
    with open(path, encoding='utf8') as f:
        return [{'state': row['text'], 'label': row['category']} for row in csv.DictReader(f)]


def clinc150_to_records(path, split='train'):
    d = json.load(open(path, encoding='utf8'))
    return [{'state': x[0], 'label': x[1]} for x in d[split]]


def make_decision_example(r, labels):
    return {'state': r['state'], 'questions': [{'id': 'intent', 'type': 'choice', 'instruction': 'Which intent best matches this state?', 'options': labels}], 'ground_truth': [{'id': 'intent', 'type': 'choice', 'target': labels.index(r['label'])}]}


_NOUL_SECURITY_INTENTS = {
    'card_payment_not_recognised',
    'cash_withdrawal_not_recognised',
    'direct_debit_payment_not_recognised',
    'compromised_card',
    'lost_or_stolen_card',
    'lost_or_stolen_phone',
}

def _noul_target(label):
    return int(label.lower() in _NOUL_SECURITY_INTENTS)


def make_multi_question_example(r, labels):
    label = r['label'].lower()
    route = 'card' if 'card' in label else 'transfer' if 'transfer' in label else 'cash' if 'cash' in label or 'withdraw' in label else 'account' if 'account' in label or 'balance' in label else 'payment' if 'payment' in label or 'bill' in label else 'other'
    risk = 'high' if any(k in label for k in ('fraud', 'stolen', 'cash_withdrawal', 'cash withdrawal', 'lost')) else 'medium' if any(k in label for k in ('declined', 'failed', 'charge', 'wrong', 'fee', 'dispute')) else 'low'
    routes = ['account', 'card', 'cash', 'payment', 'transfer', 'other']
    risks = ['low', 'medium', 'high']
    noul = _noul_target(label)
    return {'state': r['state'], 'questions': [
        {'id': 'intent', 'type': 'choice', 'instruction': 'Which intent best matches this state?', 'options': labels},
        {'id': 'route', 'type': 'choice', 'instruction': 'Which operational route should handle this request?', 'options': routes},
        {'id': 'risk', 'type': 'score', 'instruction': 'What risk level best describes this request?', 'levels': risks},
        {'id': 'security', 'type': 'noul', 'instruction': 'Does this request indicate a fraud or security-related issue?'}],
        'ground_truth': [
        {'id': 'intent', 'type': 'choice', 'target': labels.index(r['label'])},
        {'id': 'route', 'type': 'choice', 'target': routes.index(route)},
        {'id': 'risk', 'type': 'score', 'target': risks.index(risk)},
        {'id': 'security', 'type': 'noul', 'target': noul}]}


def canonical_question_prompt(state, q):
    if q['type'] == 'choice':
        opts = q['options']
    elif q['type'] == 'score':
        opts = q['levels']
    else:
        opts = ['yes', 'no']
    return f"STATE: {state} QUESTION: {q['instruction']} OPTIONS: {' | '.join(opts)} ANSWER:"


def canonical_answer(q, target):
    if q['type'] == 'choice':
        return q['options'][target]
    if q['type'] == 'score':
        return q['levels'][target]
    return 'yes' if target else 'no'


def serialize_for_llm(ex):
    parts = []
    for q, t in zip(ex['questions'], ex['ground_truth']):
        parts.append(canonical_question_prompt(ex['state'], q) + ' ' + canonical_answer(q, t['target']))
    return ' '.join(parts)


def serialize_prompt(ex):
    return ' '.join(canonical_question_prompt(ex['state'], q) for q in ex['questions'])


def tokenizer_texts_for_examples(examples):
    texts = []
    for ex in examples:
        texts.append(ex['state'])
        for q, t in zip(ex['questions'], ex['ground_truth']):
            texts.append(q['instruction'])
            if q['type'] == 'choice':
                texts.extend(q['options'])
            elif q['type'] == 'score':
                texts.extend(q['levels'])
            else:
                texts.extend(['yes', 'no'])
            # The prompt format is part of the model input, so its literal
            # control tokens must exist in the vocabulary rather than becoming
            # <unk>.  The answer is included here as well because it is the
            # supervised continuation.
            texts.append(canonical_question_prompt(ex['state'], q))
            texts.append(canonical_answer(q, t['target']))
    return texts


def deterministic_split(records, seed=42, ratios=(.8, .1, .1)):
    r = records[:]
    random.Random(seed).shuffle(r)
    n = len(r)
    a = int(n * ratios[0])
    b = int(n * (ratios[0] + ratios[1]))
    return r[:a], r[a:b], r[b:]


def shifted_records(records):
    return [{'state': r['state'].replace('card', 'bank card').replace('transfer', 'money transfer'), 'label': r['label']} for r in records]
