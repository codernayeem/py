''' Good students always try to solve exercise on their own first and then look at the ready made solution
    I know you are an awesome student !! :)
    Hence you will look into this code only after you have done your due diligence.
    If you are not an awesome student who is full of laziness then only you will come here
    without writing single line of code on your own. In that case anyways you are going to
    face my anger with fire and fury !!!
'''

import math, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

def predict_using_sklean():
    df = pd.read_csv("test_scores.csv")
    r = LinearRegression()
    r.fit(df[['math']], df.cs)
    return r.coef_, r.intercept_

def gradient_descent(x,y):
    m_curr, b_curr = 0, 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.0002

    cost_previous = 0

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1 / n) * sum([value ** 2 for value in (y - y_predicted)])
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print ("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost, i))

    return m_curr, b_curr

if __name__ == "__main__":
    df = pd.read_csv("test_scores.csv")
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = gradient_descent(x, y)
    print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

    m_sklearn, b_sklearn = predict_using_sklean()
    print("Using sklearn: Coef {} Intercept {}".format(m_sklearn, b_sklearn))


# Result : 
# Using gradient descent function: Coef 1.0177381667350405 Intercept 1.9150826165722297
# Using sklearn: Coef [1.01773624] Intercept 1.9152193111569034
