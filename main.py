import os
from src.inception_model import InceptionModel
from src.cae_model import CAE
from src.clustering import ClusterAlgorithm
# from src.evaluation import Evaluation


import argparse
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test pretrained model.')
    parser.add_argument('--inception model', dest='inception_model',
                        default=None, type=str)

    parser.add_argument('--autoencoder', dest='pretrained_autoencoder',
                        default=None, type=str)

    if len(sys.argv)>1:
        print("Only one model can be run at the same time")
        exit(-1)

    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    if args.inception_mode is None:
        if args.pretrained_autoencoder is None:
            model = CAE(args.pretrained_autoencoder)
        else:
            model = CAE(os.path.join(os.getcwd(), args.pretrained_autoencoder))
    else:
        model=InceptionModel(os.path.join(os.getcwd(), args.inception_model))

    # embeddings vraca kao Ordered dict kako je ucitavao i tako redom idu u clusteirng
    embeddings_dict = model.extract_embeddings()
    # promijenjeno s iteritems u items
    image_names, latent_representations = zip(*embeddings_dict.items())

    cluster = ClusterAlgorithm('KMeans', image_names, latent_representations, c=10)
    labels = cluster.fit()

    # writing Cae clustering resulting labels into file
    output = open("labels.txt", "w")
    for index, image_name in enumerate(image_names):
        output.write("Image " + image_name + ", " + str(labels[index]))
        output.write("\n")
    output.close()

    # model_name = 'KMeans'
    # k_range = [7, 8, 9, 10, 11, 12, 13, 14, 15]
    # evaluation = Evaluation(model_name, inception_img_names, inception_latent_representations, k_range)
    # evaluation.evaluate()


if __name__ == '__main__':
    main()
