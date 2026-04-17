import numpy as np

from data import build_example


class _FakeTokenizer:
    """Deterministic tokenizer: one char = one token, codepoint is the id.
    Good enough for mask-correctness tests without pulling in `transformers`.
    """

    pad_token_id = 0
    eos_token_id = 1  # SOH — not a real character anyone would use

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]


def test_completion_mask_covers_only_response_tokens():
    tok = _FakeTokenizer()
    prompt = "Q: hi"      # 5 chars
    chosen = "A: yo"      # 5 chars -> 5 chars + 1 EOS = 6
    rejected = "A: nope"  # 7 chars -> 7 chars + 1 EOS = 8
    max_len = 32

    ex = build_example(tok, prompt, chosen, rejected, max_len)

    # Chosen: prompt = 5 tokens, response (with EOS) = 6 tokens, pad = 21.
    assert ex["chosen_attention_mask"].sum() == 5 + 6
    assert ex["chosen_completion_mask"].sum() == 6
    # Completion mask zero on prompt region
    assert ex["chosen_completion_mask"][:5].sum() == 0
    # Completion mask zero on pad region
    assert ex["chosen_completion_mask"][11:].sum() == 0
    # Completion mask is contiguous on response
    np.testing.assert_array_equal(ex["chosen_completion_mask"][5:11], np.ones(6, dtype=np.int32))

    # Rejected: prompt = 5, response+EOS = 8, pad = 19
    assert ex["rejected_attention_mask"].sum() == 5 + 8
    assert ex["rejected_completion_mask"].sum() == 8
    assert ex["rejected_completion_mask"][:5].sum() == 0
    assert ex["rejected_completion_mask"][13:].sum() == 0


def test_completion_mask_disjoint_from_pad():
    tok = _FakeTokenizer()
    ex = build_example(tok, "p", "r", "r", max_length=16)
    # (attention_mask == 0) and (completion_mask == 1) can never both hold.
    pad_region = ex["chosen_attention_mask"] == 0
    cmask = ex["chosen_completion_mask"] == 1
    assert not np.any(pad_region & cmask)


def test_truncation_preserves_completion():
    tok = _FakeTokenizer()
    # Prompt + completion > max_length. Prompt must be truncated, completion kept.
    prompt = "x" * 20
    chosen = "yyyy"        # 4 chars + 1 EOS = 5
    rejected = "yyyy"
    max_len = 10

    ex = build_example(tok, prompt, chosen, rejected, max_len)
    # Full completion (5 tokens) must survive in both cases.
    assert ex["chosen_completion_mask"].sum() == 5
    assert ex["rejected_completion_mask"].sum() == 5
    # No pad inside attention mask — truncation filled the budget.
    assert ex["chosen_attention_mask"].sum() == max_len
