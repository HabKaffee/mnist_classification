# Lecun format to csv was taken from https://github.com/egcode/MNIST-to-CSV
import pandas as pd

def convert_lecun_to_csv(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for _ in range(n):
        image = [ord(l.read(1))]
        for _ in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

def preprocess_data_to_csv():
    convert_lecun_to_csv("data/train-images.idx3-ubyte", "data/train-labels-idx1-ubyte/train-labels-idx1-ubyte",
            "data/mnist_train.csv", 60000)
    convert_lecun_to_csv("data/t10k-images.idx3-ubyte", "data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
            "data/mnist_test.csv", 10000)

    df_orig_train = pd.read_csv('data/mnist_train.csv')
    df_orig_test = pd.read_csv('data/mnist_test.csv')
    
    df_orig_train.rename(columns={'5':'label'}, inplace=True)
    df_orig_test.rename(columns={'7':'label'}, inplace=True)
    
    df_orig_train.to_csv('data/mnist_train_final.csv', index=False)
    df_orig_test.to_csv('data/mnist_test_final.csv', index=False)
    


