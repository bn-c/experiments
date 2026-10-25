from ebm.dataset.loader import normalize_board


def test_normalize_board_dots_to_zeros():
    raw = ('.' * 40) + ('1' * 41)
    out = normalize_board(raw)
    assert len(out) == 81
    assert out.count('0') == 40
    assert '.' not in out


def test_normalize_board_rejects_bad_length():
    try:
        normalize_board('123')
        raise AssertionError('expected ValueError')
    except ValueError:
        pass
