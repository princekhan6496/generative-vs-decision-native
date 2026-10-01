import torch
from datasets.preprocessing import canonical_question_prompt, canonical_answer, make_multi_question_example
from experiments.benchmark import llm_choice_probs
from mini_llm.tokenizer import SimpleTokenizer
from mini_llm.model import MiniLLM


def test_canonical_prompt_is_shared_by_training_and_evaluation():
    ex = make_multi_question_example(
        {'state': 'my card was stolen', 'label': 'cash_withdrawal'},
        ['cash_withdrawal', 'card_arrival'],
    )
    q, t = ex['questions'][0], ex['ground_truth'][0]
    prompt = canonical_question_prompt(ex['state'], q)
    training_text = prompt + ' ' + canonical_answer(q, t['target'])
    assert prompt == 'STATE: my card was stolen QUESTION: Which intent best matches this state? OPTIONS: cash_withdrawal | card_arrival ANSWER:'
    assert training_text.startswith(prompt + ' ')
    assert 'state:' not in prompt and 'question ' not in prompt and 'answer:' not in prompt


def test_candidate_scoring_has_no_eos_between_answer_and_candidate():
    tok = SimpleTokenizer().fit(['STATE: QUESTION: OPTIONS: ANSWER:', 'red', 'blue'])
    model = MiniLLM(tok.vocab_size, d_model=32, n_heads=4, n_layers=1, d_ff=64, max_seq_len=32)
    captured = {}
    original_forward = model.forward

    def wrapped(x):
        captured['ids'] = x.detach().cpu().tolist()[0]
        return original_forward(x)

    model.forward = wrapped
    prompt = 'STATE: item QUESTION: choose color OPTIONS: red | blue ANSWER:'
    llm_choice_probs(model, tok, prompt, ['red', 'blue'], torch.device('cpu'))
    ids = captured['ids']
    answer_id = tok.stoi['answer']
    colon_id = tok.stoi[':']
    eos_id = tok.stoi['<eos>']
    candidate_ids={tok.stoi['red'], tok.stoi['blue']}
    pos = next(i for i in range(len(ids)-1) if ids[i] == answer_id and ids[i+1] == colon_id)
    assert ids[pos+2] in candidate_ids
    assert ids[pos+2] != eos_id
