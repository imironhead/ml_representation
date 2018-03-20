# Siamese Neural Networks for One-shot Image Recognition

* Did not fully replicate the result (only a little part of them per my imagination).
* Use Adam,
* No regularization.
* omniglot dataset is split into 4 non-overlapped parts:
  1. alphabet T + drawer T
  2. alphabet V + drawer V
  3. alphabet T + drawer V
  4. alphabet V + drawer T
* alphabet T + drawer T are used for training.
* alphabet V + drawer V are used for validation.
* Validation accuracy > 90%
* I did not want to build a small model for one-shot learning for MNIST, so scaled up some MNIST digits for testing, but the result was very bad.
* An interesting thing in the model: if 2 duplicated images are feed to the siamese network, the probability of same image is 0.5 (because both image flow through the same network and use L1 distance to joint, which result in a zero vector)
