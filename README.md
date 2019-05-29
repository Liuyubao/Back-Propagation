# Back-Propagation

Implemented the back propagation algorithm

## 1.Main codes
```python

def train_model(model, train_ys, train_xs, dev_ys, dev_xs, args):
    N = len(train_xs)
    epoch = N/5
    bestModelForDev = model[0]
    bestAccuracy = 0
    w1,w2 = extract_weights(model)
    for i in range(args.iterations):
        for k in range(N):
            ## forward pass ##
            h_in = np.dot(model[0],train_xs[k])  #(5,1)
            h_out = sigmoid(h_in)
            h_out = np.append(h_out,1)   #(6,1)
            out_in = np.dot(model[1],h_out)   #(1,)
            p_out = sigmoid(out_in)     #(1,)
            ## backward pass ##
            delta_w2 = np.dot(-(train_ys[k]-p_out),p_out*(1-p_out))  #(1,)
            dEd_w2 = delta_w2*h_out          #(1,)
            delta_w1 = np.dot(model[1]*delta_w2,h_out*(1-h_out))   #(1,1)
            dEd_w1 = delta_w1*train_xs[k]    #(124,1)
            dEd_w1 = dEd_w1.T          #(1,124)
            model[1] = model[1]-args.lr*dEd_w2   #(1,6)
            model[0] = model[0]-args.lr*dEd_w1   #(5,124)
            if not args.nodev and k % epoch == 0:
                currentAccuracy = test_accuracy(model, dev_ys, dev_xs)
                if currentAccuracy > bestAccuracy:
                    bestAccuracy = currentAccuracy
                    bestModelForDev = model
    if not args.nodev:
        return bestModelForDev
    return model



```


## 2.Performance changes with number of iterations increases(All other params with default values)

accuracy/Number of iterations:

0.7567663396761612 / 1

0.7567663396761612 / 2

0.7567663396761612 / 4

0.7567663396761612  / 8

0.8488358350076823 / 16




## 3.How to best use dev data during training.

Here I added a variable "epoch" whose value is N/5. For each epoch, I will call test_accuracy to check the goodness of the weights. If the current epoch's accuracy is bigger than the best, I will updated it using current's weights.


## 4.Performance changes with learning rate(All other params with default values)

accuracy/learning rate:

0.24323366032383878 / 0.000001

0.7567663396761612 / 0.0001

0.7567663396761612 / 0.01

0.7567663396761612 / 0.05

0.7567663396761612 / 0.1

0.8409171492731355 / 0.5

0.8438718827561754 / 1

0.8319347594846945 / 2

0.8144427372650986 / 5

## 5.Performance changes with hidden layer size (All other params with default values)

accuracy/hidden_dim:

0.849781349722255 / 1

0.7567663396761612 / 2

0.8485994563290391 / 3

0.7567663396761612 / 5

0.7567663396761612 / 10

0.24323366032383878 / 50

0.24323366032383878 / 100

## 6.Using the initial weights in w1.txt and w2.txt

Accuracy: 0.8494267817042903


## 7.Steps

Here I chose to add a bias=1 after the sigmoid of the weights in current layer

## 8.Small Conclusion:

From the above experiements, I found that:

8a)Basically the accuracy will increase with the number of iterations increase. It should be under the condition that not overfitting;

8b)With learning rate too small will leads to overfitting;

8c)The big number of the hidden_dim will not lead to perfect result. However, it will give me a overfitting result.

8d)I think if I try with more layers or more complicated NN like CNN, the result will be much better.
