import os,sys
from decoder import Decoder
from net_properties import NetProperties
import matplotlib.pyplot as plt
from network import Network
from utils import Vocab
import pickle

class DepModel:
    def __init__(self, model_path, vocab, net_properties):
        '''
            You can add more arguments for examples actions and model paths.
            You need to load your model here.
            actions: provides indices for actions.
            it has the same order as the data/vocabs.actions file.
        '''

        # if you prefer to have your own index for actions, change this.
        self.actions = ['SHIFT', 'LEFT-ARC:rroot', 'LEFT-ARC:cc', 'LEFT-ARC:number',
                        'LEFT-ARC:ccomp', 'LEFT-ARC:possessive', 'LEFT-ARC:prt',
                        'LEFT-ARC:num', 'LEFT-ARC:nsubjpass', 'LEFT-ARC:csubj',
                        'LEFT-ARC:conj', 'LEFT-ARC:dobj', 'LEFT-ARC:nn', 'LEFT-ARC:neg',
                        'LEFT-ARC:discourse', 'LEFT-ARC:mark', 'LEFT-ARC:auxpass',
                        'LEFT-ARC:infmod', 'LEFT-ARC:mwe', 'LEFT-ARC:advcl', 'LEFT-ARC:aux',
                        'LEFT-ARC:prep', 'LEFT-ARC:parataxis', 'LEFT-ARC:nsubj',
                        'LEFT-ARC:<null>', 'LEFT-ARC:rcmod', 'LEFT-ARC:advmod',
                        'LEFT-ARC:punct', 'LEFT-ARC:quantmod', 'LEFT-ARC:tmod',
                        'LEFT-ARC:acomp', 'LEFT-ARC:pcomp', 'LEFT-ARC:poss',
                        'LEFT-ARC:npadvmod', 'LEFT-ARC:xcomp', 'LEFT-ARC:cop',
                        'LEFT-ARC:partmod', 'LEFT-ARC:dep', 'LEFT-ARC:appos',
                        'LEFT-ARC:det', 'LEFT-ARC:amod', 'LEFT-ARC:pobj', 'LEFT-ARC:iobj',
                        'LEFT-ARC:expl', 'LEFT-ARC:predet', 'LEFT-ARC:preconj',
                        'LEFT-ARC:root', 'RIGHT-ARC:rroot', 'RIGHT-ARC:cc', 'RIGHT-ARC:number',
                        'RIGHT-ARC:ccomp', 'RIGHT-ARC:possessive', 'RIGHT-ARC:prt',
                        'RIGHT-ARC:num', 'RIGHT-ARC:nsubjpass', 'RIGHT-ARC:csubj',
                        'RIGHT-ARC:conj', 'RIGHT-ARC:dobj', 'RIGHT-ARC:nn', 'RIGHT-ARC:neg',
                        'RIGHT-ARC:discourse', 'RIGHT-ARC:mark', 'RIGHT-ARC:auxpass',
                        'RIGHT-ARC:infmod', 'RIGHT-ARC:mwe', 'RIGHT-ARC:advcl',
                        'RIGHT-ARC:aux', 'RIGHT-ARC:prep', 'RIGHT-ARC:parataxis',
                        'RIGHT-ARC:nsubj', 'RIGHT-ARC:<null>', 'RIGHT-ARC:rcmod',
                        'RIGHT-ARC:advmod', 'RIGHT-ARC:punct', 'RIGHT-ARC:quantmod',
                        'RIGHT-ARC:tmod', 'RIGHT-ARC:acomp', 'RIGHT-ARC:pcomp',
                        'RIGHT-ARC:poss', 'RIGHT-ARC:npadvmod', 'RIGHT-ARC:xcomp',
                        'RIGHT-ARC:cop', 'RIGHT-ARC:partmod', 'RIGHT-ARC:dep',
                        'RIGHT-ARC:appos', 'RIGHT-ARC:det', 'RIGHT-ARC:amod',
                        'RIGHT-ARC:pobj', 'RIGHT-ARC:iobj', 'RIGHT-ARC:expl',
                        'RIGHT-ARC:predet', 'RIGHT-ARC:preconj', 'RIGHT-ARC:root']

        self.model = Network(vocab, net_properties)
        self.model.load(model_path)


    def score(self, str_features):
        '''
        :param str_features: String features
        20 first: words, next 20: pos, next 12: dependency labels.
        DO NOT ADD ANY ARGUMENTS TO THIS FUNCTION.
        :return: list of scores
        '''
        # change this part of the code.

        return network.decode(str_features)

if __name__=='__main__':
    # define word, pos, and label embeddings
    # define hidden layer and minibatch size
    we, pe, le, h1, h2, mb = 64, 32, 32, 200, 200, 1000
    net_properties = NetProperties(we, pe, le, h1, h2, mb)
    # define epochs
    epochs = 7

    data_file  = "data/train.data"
    vocab_path = "data/vocab_path"

    input_p = os.path.abspath(sys.argv[1])
    output_p = os.path.abspath(sys.argv[2])
    model_path = os.path.abspath(sys.argv[3])

    # creating vocabulary file
    # already created with train_file
    vocab = Vocab()

    # writing properties and vocabulary file into pickle
    pickle.dump((vocab, net_properties), open(vocab_path, 'w'))

    # constructing network
    network = Network(vocab, net_properties)

    # training
    network.train(data_file, epochs)

    # saving network
    network.save(model_path)

    m = DepModel(model_path, vocab, net_properties)
    Decoder(m.score, m.actions).parse(input_p, output_p)