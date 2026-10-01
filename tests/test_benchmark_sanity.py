import numpy as np
from datasets.preprocessing import _noul_target
from experiments.benchmark import calibrate, parse_structured_option
from mini_llm.tokenizer import SimpleTokenizer


def test_noul_mapping_has_semantic_positive_and_negative_labels():
    positive = ['compromised_card', 'card_payment_not_recognised', 'lost_or_stolen_card']
    negative = ['cash_withdrawal', 'card_arrival', 'cash_withdrawal_charge']
    assert all(_noul_target(x) == 1 for x in positive)
    assert all(_noul_target(x) == 0 for x in negative)
    assert len({_noul_target(x) for x in positive + negative}) == 2


def test_calibration_skips_small_validation_and_invalid_probabilities():
    probs = np.array([[.9, .1], [.2, .8]])
    targets = np.array([0, 1])
    _, temperature, status = calibrate(probs, targets)
    assert temperature is None
    assert status == 'skipped_insufficient_validation_data'
    bad = np.array([[np.nan, 1.0], [.2, .8]])
    _, temperature, status = calibrate(bad, np.array([0, 1]), min_examples=2)
    assert temperature is None
    assert status == 'skipped_invalid_validation_probabilities'


def test_structured_parser_requires_complete_option():
    tok = SimpleTokenizer().fit(['ANSWER:', 'cash withdrawal', 'cash'])
    assert parse_structured_option('STATE: x ANSWER: cash withdrawal', tok, ['cash', 'cash withdrawal']) == 1
    assert parse_structured_option('STATE: x ANSWER: cash withdrawal extra', tok, ['cash', 'cash withdrawal']) is None


def test_benchmark_sanity_rejects_empty_validation_when_calibration_requested():
    from experiments.benchmark import _sanity_check_examples
    import pytest
    with pytest.raises(ValueError):
        _sanity_check_examples([{'label': 'a'}], [], [{'label': 'a'}], {'a'})


def test_candidate_probabilities_are_finite_and_normalized():
    import torch
    from experiments.benchmark import llm_choice_probs
    from mini_llm.model import MiniLLM
    tok = SimpleTokenizer().fit(['STATE:', 'QUESTION:', 'OPTIONS:', 'ANSWER:', 'red', 'blue'])
    model = MiniLLM(tok.vocab_size, d_model=32, n_heads=4, n_layers=1, d_ff=64, max_seq_len=32)
    q = {'type': 'choice', 'instruction': 'choose color', 'options': ['red', 'blue']}
    p = llm_choice_probs(model, tok, 'STATE: item QUESTION: choose color OPTIONS: red | blue ANSWER:', ['red', 'blue'], torch.device('cpu'), q=q, state='item')
    assert np.isfinite(p).all()
    assert np.isclose(p.sum(), 1.0)

def test_llm_training_masks_prompt_but_keeps_answer_and_eos():
    from datasets.preprocessing import make_multi_question_example
    from experiments.benchmark import _llm_training_sequence
    from mini_llm.tokenizer import SimpleTokenizer
    from mini_llm.model import MiniLLM
    ex = make_multi_question_example({'state':'cash withdrawal issue','label':'cash_withdrawal'}, ['cash_withdrawal','card_arrival'])
    tok = SimpleTokenizer().fit(['cash withdrawal issue','cash_withdrawal','card_arrival'] + [q['instruction'] for q in ex['questions']] + ['yes','no','low','medium','high'])
    model = MiniLLM(tok.vocab_size, d_model=16, n_heads=4, n_layers=1, d_ff=32, max_seq_len=64)
    ids, labels, prompt_ids, answer_ids = _llm_training_sequence(tok, model, ex, ex['questions'][0], ex['ground_truth'][0])
    assert labels[:len(prompt_ids)] == [-100] * len(prompt_ids)
    assert labels[len(prompt_ids):len(prompt_ids)+len(answer_ids)] == answer_ids
    assert labels[len(prompt_ids)+len(answer_ids)] == tok.stoi['<eos>']
    assert ids[len(prompt_ids):len(prompt_ids)+len(answer_ids)] == answer_ids


def test_causal_lm_shift_predicts_next_token_and_masks_prompt():
    import torch
    from mini_llm.loss import causal_lm_loss
    # Position 0 is prompt; positions 1 and 2 are supervised answer tokens.
    vocab = 5
    targets = torch.tensor([[0, 3, 4]])
    logits = torch.full((1, 3, vocab), -20.0)
    logits[0, 0, 3] = 20.0  # predicts target at position 1
    logits[0, 1, 4] = 20.0  # predicts target at position 2
    labels = torch.tensor([[-100, 3, 4]])
    loss = causal_lm_loss(logits, labels, pad_id=-100)
    assert float(loss) < 1e-5


def test_llm_training_supervises_complete_multi_token_answer_before_eos():
    from datasets.preprocessing import make_multi_question_example
    from experiments.benchmark import _llm_training_sequence
    from mini_llm.tokenizer import SimpleTokenizer
    from mini_llm.model import MiniLLM
    ex = make_multi_question_example(
        {'state': 'cash issue', 'label': 'cash_withdrawal'},
        ['cash_withdrawal', 'cash withdrawal charge'],
    )
    # Force a multi-token option for this contract test.
    ex['questions'][0]['options'][1] = 'cash withdrawal charge'
    ex['ground_truth'][0]['target'] = 1
    tok = SimpleTokenizer().fit([
        'STATE: cash issue QUESTION: Which intent best matches this state? '
        'OPTIONS: cash_withdrawal | cash withdrawal charge ANSWER:',
        'cash_withdrawal', 'cash withdrawal charge',
    ])
    model = MiniLLM(tok.vocab_size, d_model=16, n_heads=4, n_layers=1, d_ff=32, max_seq_len=64)
    ids, labels, prompt_ids, answer_ids = _llm_training_sequence(
        tok, model, ex, ex['questions'][0], ex['ground_truth'][0]
    )
    assert len(answer_ids) == 3
    assert labels[len(prompt_ids):len(prompt_ids)+len(answer_ids)] == answer_ids
    assert labels[len(prompt_ids)+len(answer_ids)] == tok.stoi['<eos>']
    assert ids[len(prompt_ids):len(prompt_ids)+len(answer_ids)+1] == answer_ids + [tok.stoi['<eos>']]


def test_prompt_control_tokens_are_in_mini_llm_vocabulary():
    from datasets.preprocessing import make_multi_question_example
    from experiments.benchmark import make_tokenizer
    ex = make_multi_question_example(
        {'state': 'waiting for my card', 'label': 'card_arrival'},
        ['card_arrival', 'lost_or_stolen_card'],
    )
    tok = make_tokenizer([ex])
    assert all(tok.stoi.get(x) is not None for x in ('state', 'question', 'options', 'answer', ':', '|'))
