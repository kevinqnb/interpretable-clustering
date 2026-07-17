import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from intercluster.rules import LinearCondition


def test_axis_aligned():
    cond = LinearCondition(
        features = np.array([1]),
        weights = np.array([1]),
        threshold = 2,
        direction = -1
    )
    
    X = np.array([
        [0, 3],
        [np.inf, 0],
        [1,1],
        [3,4]
    ])
    
    assert np.array_equal(cond.evaluate(X), np.array([False, True, True, False]))
    
def test_linear(): 
    cond = LinearCondition(
        features = np.array([0,1]),
        weights = np.array([1/2, 3]),
        threshold = 5,
        direction = 1
    )
    
    X = np.array([
        [0, 3],
        [np.inf, 0],
        [1,1],
        [3,4]
    ])
    
    assert np.array_equal(cond.evaluate(X), np.array([True, True, False, True]))
    
    
def test_with_equality():
    cond = LinearCondition(
        features = np.array([1]),
        weights = np.array([1]),
        threshold = 2,
        direction = -1
    )
    
    X = np.array([
        [0, 3],
        [np.inf, 2],
        [1,2],
        [3,4]
    ])
    
    assert np.array_equal(cond.evaluate(X), np.array([False, True, True, False]))
    
    
def test_one_point():
    cond = LinearCondition(
        features = np.array([4]),
        weights = np.array([1]),
        threshold = 2,
        direction = -1
    )
    
    X = np.array([[0,3,1000, 5, 1, 35]])
    assert np.array_equal(cond.evaluate(X), np.array([True]))



def test_display():
    X = np.array([
        [1,5,2],
        [7,3,6]
    ])
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    Xstandard = standard_scaler.fit_transform(X)
    Xminmax = minmax_scaler.fit_transform(X)
    feature_labels = ["one", "two", "three"]

    # Test some basic printing of conditions
    cond1 = LinearCondition(
        features = np.array([0, 1]),
        weights = np.array([2, 1]),
        threshold = 2,
        direction = -1
    )

    cond2 = LinearCondition(
        features = np.array([1, 2]),
        weights = np.array([3, 4]),
        threshold = 6,
        direction = 1
    )

    assert cond1.display() == r"2$\cdot$$x_0$ $+$" + "\n" + r"$x_1$" + "\n" + r"$\leq$ 2"
    assert cond2.display() == r"3$\cdot$$x_1$ $+$" + "\n" + r"4$\cdot$$x_2$" + "\n" + r"$>$ 6"

    # Test with standard scaling
    cond3 = LinearCondition(
        features = np.array([0]),
        weights = np.array([1]),
        threshold = 1,
        direction = 1
    )
    cond4 = LinearCondition(
        features = np.array([0]),
        weights = np.array([1]),
        threshold = -1,
        direction = -1
    )
    assert cond3.display(scaler=standard_scaler) == r"$x_0$" + "\n" + r"$>$ 7.0"
    assert cond4.display(scaler=standard_scaler) == r"$x_0$" + "\n" + r"$\leq$ 1.0"

    # Test with MinMax Scaling
    cond5 = LinearCondition(
        features = np.array([1]),
        weights = np.array([1]),
        threshold = 1,
        direction = 1
    )
    cond6 = LinearCondition(
        features = np.array([1]),
        weights = np.array([1]),
        threshold = 0,
        direction = -1
    )
    assert cond5.display(scaler=minmax_scaler) == r"$x_1$" + "\n" + r"$>$ 5.0"
    assert cond6.display(scaler=minmax_scaler) == r"$x_1$" + "\n" + r"$\leq$ 3.0"

    # Test with scaling and feature labels
    cond7 = LinearCondition(
        features = np.array([1, 2]),
        weights = np.array([1, 1]),
        threshold = -2,
        direction = -1
    )
    cond8 = LinearCondition(
        features = np.array([1, 2]),
        weights = np.array([1, 1]),
        threshold = 0,
        direction = -1
    )

    assert cond7.display(
        scaler=standard_scaler,
        feature_labels=feature_labels
    ) == r"two $+$" + "\n" + r"0.5$\cdot$three" + "\n" + r"$\leq$ 4.0"

    assert cond8.display(
        scaler=minmax_scaler,
        feature_labels=feature_labels
    ) == r"0.5$\cdot$two $+$" + "\n" + r"0.25$\cdot$three" + "\n" + r"$\leq$ 2.0"



    