def test_n_element(random_data):
    n = random_data.nelement()
    assert n > 0
    print(n)
