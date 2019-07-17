# IJCAI2018_SSDH
This is the code for our paper: "Semantic Structure-based Unsupervised Deep Hashing" IJCAI 2018.
Author: E. Yang, C. Deng, T. Liu, W. Liu, and D. Tao.

To run the code, you should download the dataset and specifty the image path and labels in the corresponding txt file.
For example, for training dataset, all images and their labels should be given in img_train.txt:

>\path\to\train_img1.png 1 0 1 0 1

>\path\to\train_img2.png 1 0 1 0 1

>...

As you can see, the first part is the image path, and the second part is the corresponding one-hot labels.

after set the proper runing environment and datasets, our method can be evaluated by:

```
python2 main.py
```

If your find the code usfull, please cite our paper, and if you have any questions, please feel free to contact Erkun Yang (erkunyang@gmail.com)
