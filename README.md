# Netflix-Movie-Rating-Prediction
Kaggle competition course project of course CS5691 Pattern Recognition and Machine Learning at IITM (https://www.kaggle.com/c/prml19/data)

Objective
To build a recommender system model for user - movie rating prediction and to minimise the mean
squared error on the test dataset.

Given Datasets:
train.csv - dataframe with indices ‘UserID’, ‘MovieID’ and ’Ratings’
test.csv - dataframe with indices ‘UserID’ and ‘MovieID’
genome_attributes.csv - dataframe containing feature vectors for movies
genome_scores.csv - dataframe containing scores for features given in ‘genome_attributes’

Steps Followed:
MSE calculated on 20% of train set.

1. We tried the first baseline model for collaborative filtering, i.e. returning the mean value of all
the ratings given in train dataset
rui = mu + ε
MSE = 0.85

2. Then we tried the second baseline for collaborative filtering which is including movie effect in
the 1st baseline.
rui = mu + bi + ε
MSE = 0.72

3. Now including only user effect on 1st baseline.
rui = mu + bu + ε
MSE = 0.77

4. Now combining both the user effect and movie effect on 1st baseline.
rui = mu + bu + bi + ε
MSE = 0.64

5. Then we tried implementing latent factor model on 1st baseline with dimension of p and q = 2.
rui = mu + bu + bi + pu
Tqi
MSE = 0.69

6. Then we tried implementing neighbourhood model on top of the 1st baseline. But we were not
able to calculate MSE as it was taking too much time.

7. Then we applied PCA and linear regression model (for movies) on the train dataset. It showed
a significant improvement over the previous results.

We also clipped the prediction to 5 if it was coming out to be greater than 5 and to 0.5 if it was
coming out to be less than 0.5
For PCA we tried many variances
Variance 0.90 0.95 0.99
MSE 0.8409 0.8402 0.8395
We choose the value of variance to be 0.90, because all the others were not performing well
on the test dataset.
Then we tried to scale the genome scores and then applied PCA + regression but we didn’t
observe any improvement.
MSE for full test data = 0.84

8. We then combined both the PCA + linear regression model and the 3rd baseline model by
taking weighted averages, it worked well on the test data.
MSE for full test data = 0.79
Then after doing some tuning of the weights, we got that for weights 0.4(for baseline) and
0.6(for PCA + Linear Regression) we got the best MSE.
 MSE for full test data = 0.78
Then we added validation set to our data.
 MSE for full test data = 0.77
 
9. We tried to improve the latent factor model by increasing its dimensions, we tried (100, 200,
336, 500). We got good results on train set but we didn’t see any improvement on test
dataset.

10. Then we tried kernel regression with different kernels (poly and rbf) and ridge regression with
different hyperparameters.

None of this showed any improvement over the others.

11. Then we made a new model of Regression which was specific to user and combined 3rd
baseline and PCA + Regression (for movies)

This showed a significant improvement.

 MSE for full test data = 0.761
 
Result
We finally used combination of PCA + Linear regression, 3
rd baseline model and user specific
regression and got a MSE = 0.761 on 30% of the final test data.
