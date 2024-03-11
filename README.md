# Supervised vs Unsupervised Learning

## What is Machine Learning

- Ability for machine to learn without explicitly being programmed
- Supervised learning used most in the real world applications

## Terminology

- Data used to train the model is called the `training set`
- `Train` the model to `learn` from the `training set` so it can make a `predict` 
- `x` = input variable
  - Also called a `feature` or `input feature`
- `y` = output variable
  - Also called the `target variable` 
- `m` = number of training examples
- `(x, y)` = single training example
- To reference a specific row (x<sup>(i)</sup>, y<sup>(i)</sup>) = ith training example
  - i = specific row in table
  - i is an index



![Alt text](./images/10.png)


## Supervised Learning Part 1

- Most economic value created through supervised learning
- Algorithms that learn `input -> output` mappings
    - output is called the `labeled`
- Learns from being given the "right answers"
  - Is given examples that show input `x` and corresponding `output label` y
- Once it's trained, can take a brand new input and try to create the correct output
- House price prediction is a typical example of `Regression`
  - Regression = predict a number (in house price example, the price of the house would be the prediction)

## Supervised Learning Part 2

- Classification: Breast cancer detection
  - Tries to figure out if a tumor is malignant or benign
  - The output here would be benign (0) or malignant (1)
  - This is specifically *binary classification*
  - Used to predict *categories*
- Note: this is different from regression because regression tries to predict a number out of an infinite number of possibilites
  - Classification there are a set number of outcomes to be chosen from
- There can be multiple *categories* to choose from in classification
- Some terminology, *class* and *category* are used interchangeably for the output that the model produces in a classification problem

### Two or more inputs

- You can use more than 1 input value to determine the output
- Below uses age and tumor size as inputs

![Alt text](./images/1.png)

- In the below, the line is used to help decide whether a particular tumor is benign or malignant

![Alt text](./images/2.png)

## Supervised Learning Summary

![Alt text](./images/3.png)

## Unsupervised Learning Part 1

- Most widely used form after supervised learning
- Given data that isn't associated with any output labels
- Find something interesting in **unlabeled** data
- Unsupervised because we're not trying to produce some "correct" output
- Your unsupervised learning might decide that your data can belong to 2 different `clusters`
  - Used in google news
    - Groups related news together

![Alt text](./images/4.png)

- Many businesses have huge dbs of customer data
- Group customers in different ways to better serve them and understand them

![Alt text](./images/5.png)

- With clustering, the unsupervised learning algorithm will try to group data points together in certain clusters

## Unsupervised Learning Part 2

- In unsupervised learning, data has only inputs x, but not output labels y
- Job is for unsupervised learning to find structure in the data
- Clustering:
  - Group similar data points together
- Anomaly detection:
  - Find unusual data points
    - Useful for fraud detection
- Dimensionality reduction
  - Compress data using fewer numbers without losing value

## Linear Regression Model Part 1

### Linear Regression with One Variable

- Linear regression just means fitting a straight line to your data
- Probably most widely used learning model
- Below uses square feet to indicate the price of the house
  - X = size in square feet
  - Y = price of house

![Alt text](./images/6.png)


## Linear Regression Model Part 2

- Training set includes *input features* and *output targets*
- ML learning algorithm will take training set and output some output function (used to be called hypothesis)
- The job of this function is to take new data (a new set features or just one feature) and output a result
- y_hat is the prediction that the function outputs
  - Another way to say this is that y_hat is the estimate or prediction for what the real output variable y will be (which we won't know at the time for something like a house value until the house is sold)

![Alt text](./images/7.png)

- Now how do we decide what `f` is?
- So `f` will take some input x and depending on values w and b, f will output some value as a prediction (y_hat)
  - Same as `f(x)`
- w = weight
- b = bias

![Alt text](./images/8.png)

- Linear is a good way to start before going to a more complex model
- Single variable linear regression is sometimes called univariate linear regression

![Alt text](./images/9.png)

## Linear Regression with One Variable

### Cost Function

- Tells us how well the model is doing so we can try and make the model better
- `w` and `b` are called the parameters of the model
  - Parameters can be adjusted during training
  Also referred to as `coefficients` or `weights`

![Alt text](./images/10.png)

![Alt text](./images/11.png)

- How do we determine how well our model is doing? To answer that, we use a cost function
  - For known examples, we compare the predicted value, `y_hat` to the actual value `y`, and we do this for all examples
    - We take the square difference of `y_hat` and `y` for all training examples
    - The `1/2m` is for 2 reasons
      - `1/m` to take the average
      - The `2` is to make some of the later calculations a bit neater
        - This is optional
  - This is one example, there are many more types of cost functions
    - Squared error cost is the most popular for linear regression and other types of algorithms as well

![Alt text](./images/12.png)

### Cost Function Intuition

- We want to minimize the cost (`J(w, b)`)
- Below is a simplified cost function where `b = 0`
- Goal: Minimize the cost function

![Alt text](./images/13.png)

- Below is an example where we have a perfectly fit line to our training example, `w = 1`
  - Note that the cost function is 0
- In the graph on the right had side, note that `J(1) = 0` because our cost function was 0 when `w = 1`

![Alt text](./images/14.png)

- Below is an exmaple where we have `w = 0.5` which is not perfectly fit

![Alt text](./images/15.png)

- Below is an eaxmple where we have `w = 0`

![Alt text](./images/16.png)

- We can keep doing this for many many values of `w`
- By continuing to do this calculation, we can create find what the cost function J looks like
  - Each value of parameter `w` corresponds to a different fitting line in f(x)
  - So for each point on the J(w) graph, you have a corresponding line for f(x)

![Alt text](./images/17.png)

- We want to choose `w` that minimizes `J(w)`
  - Following this example, we would choose `w = 1` because `J(w) = 0` here

### Visualizing the Cost Function

- When we have both w and b (b != 0)
  - Note that this is the 3D version of the graph we had when b = 0
- As w and b are varied, get different values for the cost function `J`
  - Any point on this graph is a particular selection of w and b

![Alt text](./images/18.png)

- These 3D plots can be generated as a 2D contour plot to better visualize the cost function and to see the minimum

![Alt text](./images/19.png)

### Visualization Examples


![Alt text](./images/20.png)

- Note where w and b intersect on the cost function is far from the minimum because it's not a good fit to the training set

![Alt text](./images/21.png)

- Another example below

![Alt text](./images/22.png)

- Example with cost function close to the minimum

![Alt text](./images/23.png)

## Training the model with Gradient Descent

### Gradient Descent

- Systematic way to minimize w and b
- Used heavily in machine learning
- Gradient descent can be used to minimize any function, not just the cost function for linear regression
- Outline
  - Start out with some initial guesses for w and b
    - Doesn't matter what initial values, commonly both are set to 0
  - Keep changing w,b to reduce J(w, b) until we settle at or near a minimum
    - J is not always a parabola with a single minimum
      - Linear regression with a cost function using squared error cost function will always be a bowl shape

![Alt text](./images/24.png)

- Want to get from hill to valley as fast as possible
- If I want to take a baby step and get down hill to valley as quickly as possible, what direction do we go?
  - This repeats until we hit a minimum

![Alt text](./images/25.png)

- What if you started in a different location?
- We may end up in a different valley
- These 2 valleys are called `local minima`

![Alt text](./images/26.png)

### Implementing Gradient Descent

- Update your parameter `w` by taking the *current* value of `w` and adjusting it a small amount
  - *Alpha = learning rate*
    - Small positive number between 0 and 1
    - Controls how big of a step we take downhill during gradient descent
  - Partial derivative of `J(w,b)` with respect to `w`
    - Which direction to take step
    - Also determiens size of step we take downhill (along with learning rate)
- There is a similar formula for `b`
  - Partial derivative happens with respect to `b` in this case
- Repeat these operations until we have convergence
  - This means that `w` and `b` do not change very much
- Want to update both `w` and `b` simultaneously

![Alt text](./images/27.png)

### Gradient descent intuition

![Alt text](./images/28.png)

### Learning Rate

- Has a huge impact on efficiency of your implementation
- If chosen poorly, gradient descent may not work at all

![Alt text](./images/29.png)

- What if you've chosen a w so that you're already at a local minimum?
  - Gradient descent will leave `w` unchanged

![Alt text](./images/30.png)

- As we get closer to the local minimum, the steps become smaller because the derivative is smaller leading to a smaller change in w

![Alt text](./images/31.png)

### Gradient Descent for Linear Regression

![Alt text](./images/32.png)

- Gradient descent can lead to a local minimum instead of a global minimum
- When you use a squared error cost function, there will NEVER be multiple local minimum, there is only a single global minimum
  - This is because the squared error cost function produces a bowl shaped cost function

![Alt text](./images/33.png)

### Running Gradient Descent

![Alt text](./images/34.png)

- Now we start taking steps to move to the minimum

![Alt text](./images/35.png)

- This gradient descent process is called `batch gradient descent`
  - `Batch` means that for each step of gradient descent uses all of the training examples for each update

![Alt text](./images/36.png)

## Multiple Features

- Original linear regression we had one feature

![Alt text](./images/37.png)

- Note that a row of features is sometimes called a `row vector`
  - A vector here is basically the same thing conceptually as an array of values
  - A row vector has all of the features values for that particular example

![Alt text](./images/38.png)

- Each feature will have it's own weight, so some features will contribute more to the price than the other, for example
  - Some features may DECREASE the value of the house, like price in years

![Alt text](./images/39.png)

- The vector weights and b are the *parameters* of the model
- We can do `vector_w * vector_x` (dot product of 2 vectors) to simplify the expression
- This is called **Multiple Linear Regression**
  - Note that this is NOT multi-variate regression (that is something else)

![Alt text](./images/40.png)

### Vectorization Part 1

- Makes code shorter and makes it run more efficiently
- Enables usage of things like GPUs and modern numerical linear algebra libraries
  - *Numpy* is the most popular of these numerical linear algebra libraries
- Below are 2 examples without vectorization

![Alt text](./images/41.png)

- With vectorization
  - Makes code shorter
  - Results in code running much faster than 2 non-vectorization implementation
    - Numpy dot function uses parallel computing to make things faster

![Alt text](./images/42.png)

### Vectorization Part 2

- Vectorization scales well to large datasets and modern hardware

![Alt text](./images/43.png)

- Below is gradient descent with vectorization

![Alt text](./images/44.png)