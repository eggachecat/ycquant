from ycquant import yc_preprocessing


def test_splitter():
    yc_preprocessing.split_train_and_test("data/product_02", header=None, sep="  ")

test_splitter()