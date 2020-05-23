# Siamese_GCN
Deep learning on Siamese GCN using pytorch.<br>
Constructed two graphs for each molecule, analyzed similarity between molecules.<br>
Tested performance based on different models and data structures.<br>

****

## Process

* I tested on the test dataset and showed the performance(accuracy and auc) every time after a training epoch.<br>

* I have compared between ChebConv from dgl and common graph convolutional networks when training Siamese network, and ChebConv gave better performance.<br>

* When training Siamese_GCN, I found that it would converge faster and had less ups and downs in performance if I trained on single molecule classification for a few epochs and passed parameters(weight and bias) to the Siamese model. I could pass parameters because the weights and bias of these two models have the same shapes.<br>

* I wanted to test whether passing parameters between models would make the latter models have better performance, so I designed three phases of training. <br>
GCN_single(training on single molecule) --> GCN_hinge(training on Siamese) --> GCN_single.<br>
I used GraphConvolution from layers.py in GCN_hinge instead of ChebConv in order to pass parameters into it. Each phase has 40 epochs. After each phase, I saved the predictions on each molecule in test folder. At last, I tested the performance of each phase on ten test folders. These results were given by comparing each molecule’s prediction and its ground truth.(This process is in read.py.)<br>

* After the first phase, the performance is:<br>
accuracy: 0.8226315789473684<br>
auc: 0.9036804103185596<br>

* After the second phase, the performance is:<br>
accuracy: 0.6846710526315789<br>
auc: 0.6992536530470914<br>

* After the second phase, the performance is:<br>
accuracy: 0.7778947368421053<br>
auc: 0.8873510560941829<br>

****

## Conclusions

* Based on [Similarity Learning with Higher-Order Graph Convolutions for Brain Network Analysis](https://arxiv.org/abs/1811.02662), using Chebyshev polynomial in Siamese neural network has better performance than using common GCN. My practice had proved this.<br>

* Passing parameters is useless if two models are different, even the subtle difference. GCN_single has three layers, the first(gc1) and second(gc2) are instances of GraphConvolution from layers.py, and I can pass parameters into these two layers. The third one is a linear layer(nn.Linear), and I cannot pass parameters into this layer. If I build a GCN_single model and trained it for 40 epochs, which means this model has converged, and then build a new GCN_single model, passing parameters into gc1 and gc2, leaving linear layer initializes randomly, the performance is actually bad.<br>

* As for passing parameters between GCN_single and GCN_hinge, it is useless since they have different layers.<br>

* For this project, training on GPU is even slower than CPU. I think this is because molecules have small graph structures, GPU is not good at analyzing small graph.<br>

* GCN_hinge has two layers. The weight of first layer, whether using ChebConv or common GCN, has the shape of feature\*hidden. The weight of second layer has the shape of hidden\*n. After passing a max-pooling at last, I can get a vector of 1*n. At first the performance was always bad when I picked n=10. Then I found that when n goes smaller, performance will be better. I think this is also because the small graph structures of molecules. Molecules don't need that many features to describe them.<br>




