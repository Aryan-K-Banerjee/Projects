#include<stdio.h>
#include<stdlib.h>
#include "neuralNetwork.h"

//VARS FOR TRAINING****************************************************
//training data parameters
int num_inputs;
int num_outputs;
int num_training_sets;
    
//2d arrays for storing inputs and outputs
double **training_inputs; 
double **training_outputs;

int num_hidden_layers;
int *hidden_layer_sizes;
//*********************************************************************

//MAIN FUNCTION
//*****************************************************************
int main(int argc, char* argv[]){
    if (argc < 3) {
        fprintf(stderr, "Issue with getting files");
        return -1;
    }
    //READING DATA AND NETWORK ARCHITECTURE FROM FILES
    //**************************************************************************
    FILE *training_file = fopen(argv[1], "r");
    if (training_file == NULL) {
        fprintf(stderr,"Error opening the training file");
        return -1;
    }

    FILE *architecture_file = fopen(argv[2], "r");
    if (architecture_file == NULL) {
        fprintf(stderr,"Error opening the architecture file");
        return -1;
    }
    
    if(readTrainingData(training_file, &training_inputs, &training_outputs, &num_inputs, &num_outputs, &num_training_sets) == -1 ||
        readNetworkArchitecture(architecture_file, &num_hidden_layers, &hidden_layer_sizes) == -1){
        fclose(training_file);
        fclose(architecture_file);
    }

    printf("Inputs and Outputs:\n");
    for (int i = 0; i < num_training_sets; i++) {
        for (int j = 0; j < num_inputs; j++) {
            printf("%lf ", training_inputs[i][j]);  // Print inputs
        }
        for (int j = 0; j < num_outputs; j++) {
            printf("%lf ", training_outputs[i][j]);  // Print outputs
        }
        printf("\n");
    }
    printf("\n");
    //**************************************************************************

    // MAKING AND TRAINING NN
    //*****************************************************************+
    struct NeuralNetwork* nn = makeNeuralNetwork(num_inputs, num_outputs, num_hidden_layers, hidden_layer_sizes);
    printNeuralNetwork(nn);
    // Learning rate (controls how much weights change with each update)
    const double lr = 0.1f;
    // Array to keep track of the order of training sets (for shuffling)
    int* training_set_order = generateTrainingOrder(num_training_sets);
    // Number of epochs (iterations over the entire training set)
    int num_epochs = 10000;
    trainNeuralNetwork(nn, training_inputs, training_outputs, num_training_sets, num_epochs, lr, "sigmoid", 0);
    
    printf("\n\nFinal Weights:\n");
    printWeightsAndBiases(nn);

    //WRITING TO FILE AND READING FROM FILE
    //**************************************************************************
    FILE *file = fopen("final_weights.txt", "w");
    if (file == NULL) {
        printf("Failed to open file for writing.\n");
        return 1;
    }
    printWeightsToFile(nn, file);
    printf("reached here\n");
    fclose(file);

    FILE *file2 = fopen("final_weights.txt", "r"); 
    if (file2 == NULL) {
        return 1;
    }
    struct NeuralNetwork* newnn = readNetworkFromFile(file2);
    printNeuralNetwork(newnn);
    
    //USING NEURAL NET
    //*****************************************************************

    useNeuralNetwork(newnn);
    
    //*****************************************************************

    //FREEING DATA

    free(nn);
    free(newnn);
    fclose(file2);
    fclose(training_file);
    fclose(architecture_file);
    return 0;
}