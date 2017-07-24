from ycquant import yc_splitter


def test_splitter():
    yc_splitter.split_train_and_test("data/demo_data")

test_splitter()