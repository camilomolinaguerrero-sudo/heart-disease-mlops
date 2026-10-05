from src.common import TARGET, load_data, split


def test_carga_y_particion():
    df = load_data()
    assert df.shape == (918, 12) and df["Cholesterol"].isna().sum() == 172
    X_tr, X_te, y_tr, y_te = split(df)
    assert len(X_te) == 184 and TARGET not in X_tr
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
