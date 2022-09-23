## Training

I'm going to use Adam as optimizer, since it's the same used in [[1](https://arxiv.org/abs/1610.00087)]. The learning rate, as an hyperparameter of the network, may change the overall accuracy: I'm going to do some tests, so that I can choose the base value.

```python
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

The loss function used is the Negative Log-Likelihood funciton as, after some tests, it outperforms the  Cross-Entropy.

We do the train for `n_epoch` times; as the training goes on, we print some information about the loss function. In particular, as we have saved all the losses in a list, we can show them into a plot.

We also test the network on the `test_loader` at the end of each epoch, so that we can see the overall result (this score has no impact on the following epochs).

Some tests show us that two epochs are enough for the network to converge. 

References: [[1](https://arxiv.org/abs/1610.00087)]