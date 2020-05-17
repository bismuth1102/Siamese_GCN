'one_hot.py' and 'data_handler.py' generates 1:1 positive and negative data.

The difference between 'models_cuda.py' and 'models_cheby_cuda.py' is that the former uses GCN in 'layers.py' to do Siamese training, while the latter uses ChebConv to do it.

'train_cuda.py' uses 'model_cuda.py', while 'train_cheby_cuda.py' uses 'models_cheby_cuda.py'.

'read.py' is responsible for read data in data/res_gcn and analysis accuracy and auc on ten folders.

To run the code, just do ./cuda.sh