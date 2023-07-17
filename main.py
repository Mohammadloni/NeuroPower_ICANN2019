"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function

from evolver import Evolver
from tqdm import tqdm
import logging
from keras import backend as K
import sys
from tensorflow.python.client import device_lib
import tensorflow as tf

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO#,
    #filename='log.txt'
)



#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(non_sort_genomes):
    S=[[] for i in range(0,len(non_sort_genomes))]
    front = [[]]
    n=[0 for i in range(0,len(non_sort_genomes))]
    rank = [0 for i in range(0, len(non_sort_genomes))]

    for p in range(0,len(non_sort_genomes)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(non_sort_genomes)):
            if (non_sort_genomes[p].accuracy > non_sort_genomes[q].accuracy and non_sort_genomes[p].param < non_sort_genomes[q].param) or (non_sort_genomes[p].accuracy >= non_sort_genomes[q].accuracy and non_sort_genomes[p].param < non_sort_genomes[q].param) or (non_sort_genomes[p].accuracy > non_sort_genomes[q].accuracy and non_sort_genomes[p].param <= non_sort_genomes[q].param):
                if non_sort_genomes[q] not in S[p]:
                    S[p].append(non_sort_genomes[q])
            elif (non_sort_genomes[q].accuracy > non_sort_genomes[p].accuracy and non_sort_genomes[q].param < non_sort_genomes[p].param) or (non_sort_genomes[q].accuracy >= non_sort_genomes[p].accuracy and non_sort_genomes[q].param < non_sort_genomes[p].param) or (non_sort_genomes[q].accuracy > non_sort_genomes[p].accuracy and non_sort_genomes[q].param <= non_sort_genomes[p].param):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if non_sort_genomes[p] not in front[0]:
                front[0].append(non_sort_genomes[p])
        
    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            # for q in S[p]:
            for q in range(0,len(S)):
                n[q] =n[q] - 1
                if(n[q]==0):
                    rank[q]=i+1
                    if non_sort_genomes[q] not in Q:
                        Q.append(non_sort_genomes[q])
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

def train_genomes(genomes, dataset):
    """Train each genome.

    Args:
        networks (list): Current population of genomes
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("***train_networks(networks, dataset)***")

    pbar = tqdm(total=len(genomes))

    for genome in genomes:
        genome.train(dataset)
        genome.neurons =genome.param
        pbar.update(1)

    pbar.close()

def get_average_accuracy(genomes):
    total_accuracy = 0
    for genome in genomes:
        total_accuracy += genome.accuracy
    return total_accuracy / len(genomes)

def generate(generations, population, all_possible_genes, dataset):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evolve the population
        population (int): Number of networks in each generation
        all_possible_genes (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("***generate(generations, population, all_possible_genes, dataset)***")
    
    evolver = Evolver(all_possible_genes)
    
    genomes = evolver.create_population(population)

    # Evolve the generation.
    for i in range( generations ):

        logging.info("***Now in generation %d of %d***" % (i + 1, generations))
        # Train and get accuracy for networks/genomes.
        train_genomes(genomes, dataset)
        sort = []
        after_sort = fast_non_dominated_sort(genomes)
        for z in range(0, len(after_sort)):
            for j in range(0, len(after_sort[z])):
                sort.append(after_sort[z][j])

        # for genome in after_sort:
        error_i = []
        neurons_i = []
        for genome in sort:
            genome.print_genome()
            neurons_i.append(genome.param)
            #print(" neruronns: %d "% (genome.param))
            error_i.append(1- genome.accuracy)
            #print(" accurracy : %.4f%% "% (genome.accuracy))
            f = open(str(i) + '_Pareto_Log.txt','a')
            f.write('\n'+ '%d  :  %.4f '% (genome.param,(1- genome.accuracy)))
            f.close()




        # Get the average accuracy for this generation.
        #average_accuracy = get_average_accuracy(sort)

        # Print out the average accuracy each generation.
        #logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        #logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Evolve!
            genomes = evolver.evolve(sort)


    # Sort our final population according to performance.
    # genomes = sorted(genomes, key=lambda x: x.accuracy, reverse=True)
    for genome in genomes:
        #genome.neurons = genome.geneparam['nb_neurons'] * genome.geneparam['nb_layers']
        genome.neurons=genome.param
    sort = []
    after_sort = fast_non_dominated_sort(genomes)

    for m in range(0, len(after_sort)):
        for n in range(0, len(after_sort[m])):
            sort.append(after_sort[m][n])

    genome.print_genome()


def print_genomes(genomes):

    logging.info('-'*80)
    for genome in genomes:
        genome.print_genome()



def main():

    population = 45 # Number of networks/genomes in each generation.
    #we only need to train the new ones....
    
    ds = 4

    if   (ds == 1):
        dataset = 'mnist_mlp'
    elif (ds == 2):
        dataset = 'mnist_cnn'
    elif (ds == 3):
        dataset = 'cifar10_mlp'
    elif (ds == 4):
        dataset = 'cifar10_cnn'
    else:
        dataset = 'mnist_mlp'

    print("***Dataset:", dataset)

    if dataset == 'mnist_cnn':
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
            'nb_neurons': [16, 32, 64, 128],
            'nb_layers':  [1, 2, 3, 4 ,5],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
            'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
        }
    elif dataset == 'mnist_mlp':
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
            'nb_neurons': [16, 32, 48, 64, 96, 128, 192 ,256, 512, 768, 1024],#, 128], #, 256, 512, 768, 1024],
            'nb_layers':  [1, 2, 3, 4, 5],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
            'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
        }
    elif dataset == 'cifar10_mlp':
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
            'nb_neurons': [64, 128, 192,256, 512, 768, 1024],
            'nb_layers':  [1, 2, 3, 4, 5],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
            'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
        }
    elif dataset == 'cifar10_cnn':
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
            'nb_neurons': [16 ,28,40],
            'nb_layers':  [2,3 ],
            'activationL': ['relu', 'elu', 'tanh', 'sigmoid', 'selu', 'swish'],
            'activationR': ['relu', 'elu', 'tanh', 'sigmoid', 'selu', 'swish'],
            'optimizer': [ 'adam', 'sgd', 'adagrad', 'adamax', 'nadam']
        }
    else:
        generations = 8 # Number of times to evolve the population.
        all_possible_genes = {
            'nb_neurons': [64, 128, 256, 512, 768, 1024],
            'nb_layers':  [1, 2, 3, 4, 5],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
            'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
        }
            
    print("***Evolving for %d generations with population size = %d***" % (generations, population))

    generate(generations, population, all_possible_genes, dataset)

if __name__ == '__main__':
    #print(device_lib.list_local_devices())
    main()
