from ycquant import yc_splitter


def test_splitter():
    yc_splitter.split_train_and_test("data/product_02", header=None, sep="  ")

test_splitter()