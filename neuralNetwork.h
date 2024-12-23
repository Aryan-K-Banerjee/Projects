#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include<stdio.h>
#include<stdlib.h>
#include "activation.h"


//STRUCTS ********************************************************************
struct Neuron{
    double output;      //the output value
    double *weights;    //array of weights feeding into the neuron
    double bias;        //the bias of this neuron
    double delta;       //the error term for backpropogation
};

struct Layer{
    struct Neuron * neurons;    //array of neurons in the layer
    int num_neurons;            //the number of neurons in the layer
};

struct NeuralNetwork{
    int num_inputs;             //number of inputs
    int num_hidden_layers;      //number of hidden layers
    struct Layer * layers;      //array of layers with layers[0] being the input and the last layer being the output
    int* hidden_layer_sizes;    //array giving the number of neurons in each hidden layer
    int num_outputs;            //the number of outputs
};

//FUNCTION DECLARATIONS ********************************************************************

//neural network creation
double initWeights();
void initializeLayers(struct NeuralNetwork *nn);
void initializeWeightsAndBiases(struct NeuralNetwork *nn);
struct NeuralNetwork* makeNeuralNetwork(int num_inputs, int num_outputs, int num_hidden_layers, int *hidden_layer_sizes);

//training
void trainNeuralNetwork(struct NeuralNetwork *nn, double **training_inputs, double **training_outputs,  int num_training_sets, int num_epochs, double learning_rate, char* activation_func, int print_every);
void forwardPropagation(struct NeuralNetwork *nn, double *current_input, char* activation_func);
void backpropagation(struct NeuralNetwork *nn, double *expected_outputs, double learning_rate, char* activation_func);

//utility
int* generateTrainingOrder(int num_training_sets);
void shuffle(int *array, int n);

//reading data
int readTrainingData(FILE *file, double ***inputs, double ***outputs, int *num_inputs, int *num_outputs, int *num_training_sets);
int readNetworkArchitecture(FILE *file, int *num_hidden_layers, int **hidden_layer_sizes);

//printing
void printNeuralNetwork(struct NeuralNetwork *nn);
void printEpochResults(int epoch, int x, double *input, struct NeuralNetwork *nn, double *expected_output);
void printWeightsAndBiases(struct NeuralNetwork *nn);
void printWeightsToFile(struct NeuralNetwork *nn, FILE *file);
struct NeuralNetwork* readNetworkFromFile(FILE *file);

//using nn
void useNeuralNetwork(struct NeuralNetwork *nn);

//NEURAL NETWORK FUNCTIONS ********************************************************************

//function to initalize random weights
double initWeights(){
    return 1.0 + (((double) rand()) / ((double) RAND_MAX));  //range from 1 to 2
}

//initializing each layer of the neural network with the correct number of neurons
void initializeLayers(struct NeuralNetwork *nn) {
    int total_layers = nn->num_hidden_layers + 2; //hidden layers + input layer + output layer
    nn->layers = (struct Layer *)malloc(total_layers * sizeof(struct Layer));

    //initialize the input layer
    nn->layers[0].num_neurons = nn->num_inputs; //input layer has num_inputs neurons
    nn->layers[0].neurons = (struct Neuron *)malloc(nn->num_inputs * sizeof(struct Neuron));

    //initialize the hidden layers
    for (int i = 1; i <= nn->num_hidden_layers; i++) {
        nn->layers[i].num_neurons = nn->hidden_layer_sizes[i - 1]; //get the size of the i hidden layer
        nn->layers[i].neurons = (struct Neuron *)malloc(nn->layers[i].num_neurons * sizeof(struct Neuron));
    }

    //initialize the output layer
    int output_layer_index = nn->num_hidden_layers + 1; //the last layer
    nn->layers[output_layer_index].num_neurons = nn->num_outputs;   //has num_output neurons
    nn->layers[output_layer_index].neurons = (struct Neuron *)malloc(nn->num_outputs * sizeof(struct Neuron));
}

//function to initialize random weights and biases for all layers except the input layer
void initializeWeightsAndBiases(struct NeuralNetwork *nn) {
    //go through all layers except the input layer starting from layer 1
    for (int l = 1; l <= nn->num_hidden_layers + 1; l++) {
        struct Layer *layer = &nn->layers[l];

        //randomly nitialize weights and biases for all neurons in the layer
        for (int n = 0; n < layer->num_neurons; n++) {
            struct Neuron *neuron = &layer->neurons[n];

            //each neuron has the number of inputs to that layer weights)
            int num_weights = nn->layers[l - 1].num_neurons;  //previous layers neurons
            neuron->weights = (double *)malloc(num_weights * sizeof(double));

            //initialize weights and bias to random values
            for (int w = 0; w < num_weights; w++) {
                neuron->weights[w] = initWeights();
            }
            neuron->bias = initWeights();
        }
    }
}

//returning a neural network with the desired architecture
struct NeuralNetwork* makeNeuralNetwork(int num_inputs, int num_outputs, int num_hidden_layers, int *hidden_layer_sizes) {
    struct NeuralNetwork* nn = (struct NeuralNetwork*)malloc(sizeof(struct NeuralNetwork));
    if (nn == NULL) {
        printf("Cant make neural network\n");
        return NULL;
    }
    nn->num_inputs = num_inputs;
    nn->num_outputs = num_outputs;
    nn->num_hidden_layers = num_hidden_layers;
    nn->hidden_layer_sizes = hidden_layer_sizes;

    initializeLayers(nn);
    initializeWeightsAndBiases(nn);

    return nn;
}

//***********************************************************************************************

//TRAINING FUNCTIONS ********************************************************************

//function that trains the neural network for the desired amount of epochs
void trainNeuralNetwork(struct NeuralNetwork *nn, double **training_inputs, double **training_outputs, 
                        int num_training_sets, int num_epochs, double learning_rate, char* activation_func, int print_every) {
    //generate training order and shuffle the training order at the start of each epoch
    int *training_set_order = generateTrainingOrder(num_training_sets);

    //loop through the training for each epoch
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        //shuffle the training set order at the start of each epoch
        shuffle(training_set_order, num_training_sets);

        //loop through each training example in the shuffled order
        for (int set = 0; set < num_training_sets; set++) {
            int i = training_set_order[set];  //get the index of the current training set
            double *input = training_inputs[i];  //the current input
            double *expected_output = training_outputs[i];  //the expected output

            //do forward propagation with the current input
            forwardPropagation(nn, input, activation_func);

            //only print results if print_every is not 0 and the epoch is divisible by print_every
            if (print_every > 0 && epoch % print_every == 0) {
                printEpochResults(epoch, set, input, nn, expected_output);
            }
        
            //o backpropagation using the expected output
            backpropagation(nn, expected_output, learning_rate, activation_func);
        }
    }
    printf("TRAINING COMPLETE******************\n");
    free(training_set_order);
}


//the forward pass
void forwardPropagation(struct NeuralNetwork *nn, double *current_input, char* activation_func) {
    //set the input layer neurons to the current input
    for (int i = 0; i < nn->num_inputs; i++) {
        nn->layers[0].neurons[i].output = current_input[i];
    }

    //propagate through the hidden layers and output layer
    for (int l = 1; l <= nn->num_hidden_layers + 1; l++) {
        struct Layer *layer = &nn->layers[l];
        struct Layer *prevLayer = &nn->layers[l - 1];

        //for each neuron in the current layer compute using the previous inputs and weights
        for (int n = 0; n < layer->num_neurons; n++) {
            struct Neuron *neuron = &layer->neurons[n];
            double sum = neuron->bias;  //start with the bias
            //sum the inputs to the neuron from the previous layer
            for (int w = 0; w < prevLayer->num_neurons; w++) {
                sum += prevLayer->neurons[w].output * neuron->weights[w];
            }
            //apply the activation function to the sum
            neuron->output = activation(sum,activation_func);
        }
    }
}

//backpropogation, complex but makes sense in the end took a lot of troubleshooting but is magical now that it works
void backpropagation(struct NeuralNetwork *nn, double *expected_outputs, double learning_rate, char* activation_func) {
    //Step 1: calculate the error at the output layer and calculate the delta values for the output layer
    for (int i = 0; i < nn->num_outputs; i++) {
        struct Neuron *outputNeuron = &nn->layers[nn->num_hidden_layers+1].neurons[i];
        double target = expected_outputs[i]; //actual output for the current training example
        double predicted = outputNeuron->output;

        //calculate the error (difference between predicted and true output)
        double error = target - predicted;
        //calculate the delta using derivative of the activation/gradient descent
        outputNeuron->delta = error * dActivation(predicted,activation_func);

        //print to see if the error is being calculated correctly
        //printf("Output Neuron %d: predicted=%.4f, target=%.4f, error=%.4f, delta=%.8f\n", i, predicted, target, error, outputNeuron->delta);
    }

    //Step 2: use the future error to calculate the current error back through the hidden layers
    for (int l = nn->num_hidden_layers; l >= 1; l--) {
        struct Layer *layer = &nn->layers[l];
        struct Layer *nextLayer = &nn->layers[l + 1];

        //for each neuron in the current hidden layer
        for (int n = 0; n < layer->num_neurons; n++) {
            struct Neuron *neuron = &layer->neurons[n];
            //calculate the sum of the delta values from the next layer neurons
            double sum = 0.0;
            for (int m = 0; m < nextLayer->num_neurons; m++) {
                struct Neuron *nextNeuron = &nextLayer->neurons[m];
                double cur_weight =  nextNeuron->weights[n];
                sum += cur_weight * nextNeuron->delta; 
            }
            neuron->delta = sum * dActivation(neuron->output,activation_func);  //use derivative of activation function
        }
    }

    //Step 3: update weights and bias for all layers starting from the output layer going back
    for (int l = nn->num_hidden_layers + 1; l >= 1; l--) {
        struct Layer *layer = &nn->layers[l];
        struct Layer *prevLayer = &nn->layers[l - 1];

        //for each neuron in the current layer
        for (int n = 0; n < layer->num_neurons; n++) {
            struct Neuron *neuron = &layer->neurons[n];
            //update the bias for the neuron
            neuron->bias += learning_rate * neuron->delta;
            //update the weights for the neuron
            for (int m = 0; m < prevLayer->num_neurons; m++) {
                struct Neuron *prevNeuron = &prevLayer->neurons[m];
                neuron->weights[m] += learning_rate * neuron->delta * prevNeuron->output;
            }
        }
    }
}


//***********************************************************************************************

//UTILITY FUNCTIONS ********************************************************************
//function to make the training order based on the number of inputs
int* generateTrainingOrder(int num_training_sets) {
    //allocate memory for the training_order array
    int *training_order = (int *)malloc(num_training_sets * sizeof(int));
    if (training_order == NULL) {
        printf("Malloc failed for training order\n");
        return NULL; 
    }
    //fill the training_order array from 0 to the number of training sets
    for (int i = 0; i < num_training_sets; i++) {
        training_order[i] = i;
    }
    return training_order;
}

//function to shuffle the training order randomly
void shuffle(int *array, int n){
    if(n > 1 ){
        for(int i = 0; i < n - 1; i++){
            int j = i + rand()/ (RAND_MAX/ (n-i) + 1);
            int temp = array[j];
            array[j] = array[i];
            array[i] = temp;
        }
    }
}
//***********************************************************************************************


//READING DATA FUNCTIONS ********************************************************************
//Reading Input File Data Format
// #inputs #outputs #training sets
// inputa1 inputa2 outputa1 outputa2
// ...
// inputn1 inputn2 outputn1 outputn2
//
//Example for Xor in xor.txt and this function takes pointers to all the parameters
int readTrainingData(FILE *file, double ***inputs, double ***outputs, int *num_inputs, int *num_outputs, int *num_training_sets) {
    //reading the first line to get num_inputs, num_outputs, and num_training_sets
    if (fscanf(file, "%d %d %d", num_inputs, num_outputs, num_training_sets) != 3) {
        printf("Error reading the first line\n");
        return -1;
    }

    //dynamically allocate memory for inputs and outputs based on the size
    *inputs = (double **)malloc(*num_training_sets * sizeof(double *));
    *outputs = (double **)malloc(*num_training_sets * sizeof(double *));
    
    if (*inputs == NULL || *outputs == NULL) {
        printf("Malloc error\n");
        return -1;
    }

    //allocate memory for each input and output row
    for (int i = 0; i < *num_training_sets; i++) {
        (*inputs)[i] = (double *)malloc(*num_inputs * sizeof(double));
        (*outputs)[i] = (double *)malloc(*num_outputs * sizeof(double));

        if ((*inputs)[i] == NULL || (*outputs)[i] == NULL) {
            printf("Malloc error 2\n");
            return -1;
        }
    }

    //read the inputs and output from the file
    for (int i = 0; i < *num_training_sets; i++) {
        for (int j = 0; j < *num_inputs; j++) {
            fscanf(file, "%lf", &(*inputs)[i][j]);  //read inputs
        }
        for (int j = 0; j < *num_outputs; j++) {
            fscanf(file, "%lf", &(*outputs)[i][j]);  //read expected outputs
        }
    }
    return 0;
}

//Reading the Network Architecture from architecture file
// #hiddenlayers
// #neuronshiddenlayer1 #neuronshiddenlayer2 ...
//
//Example for Xor in xorarch.txt this function allows for flexible network architecture
int readNetworkArchitecture(FILE *file, int *num_hidden_layers, int **hidden_layer_sizes) {
    //read the number of hidden layers
    if (fscanf(file, "%d", num_hidden_layers) != 1) {
        printf("Error reading number of hidden layers\n");
        return -1;
    }

    //malloc memory for the hidden layer sizes array
    *hidden_layer_sizes = (int *)malloc(*num_hidden_layers * sizeof(int));
    if (*hidden_layer_sizes == NULL) {
        printf("Error allocating memory for hidden layers\n");
        return -1;
    }

    //read the number of neurons in each hidden layer and put in hidden layer sizes array
    for (int i = 0; i < *num_hidden_layers; i++) {
        if (fscanf(file, "%d", &(*hidden_layer_sizes)[i]) != 1) {
            printf("Error reading the size of hidden layer %d\n", i + 1);
            return -1;
        }
    }

    return 0;
}
//***********************************************************************************************


//PRINTING FUNCTIONS ********************************************************************
//function to print the neural network's structure
void printNeuralNetwork(struct NeuralNetwork *nn) {
    //print the input layer
    printf("Input Layer (%d neurons):\n", nn->num_inputs);
    for (int i = 0; i < nn->num_inputs; i++) {
        printf("Neuron %d: Output = %f\n", i, nn->layers[0].neurons[i].output);  //only output is printed
    }

    //print the hidden layers
    printf("\nHidden Layers:\n");
    for (int i = 1; i <= nn->num_hidden_layers; i++) {
        printf("Hidden Layer %d (%d neurons):\n", i, nn->layers[i].num_neurons);
        for (int j = 0; j < nn->layers[i].num_neurons; j++) {
            printf("  Neuron %d: Output = %f, Bias = %f, Weights = [", j, nn->layers[i].neurons[j].output, nn->layers[i].neurons[j].bias);
            for (int k = 0; k < nn->layers[i - 1].num_neurons; k++) {  //weights are linked to the previous layer
                printf("%lf ", nn->layers[i].neurons[j].weights[k]);
            }
            printf("]\n");
        }
    }

    //print the output layer
    printf("\nOutput Layer (%d neurons):\n", nn->num_outputs);
    for (int i = 0; i < nn->num_outputs; i++) {
        printf("  Neuron %d: Output = %f, Bias = %f, Weights = [", i, nn->layers[nn->num_hidden_layers + 1].neurons[i].output, nn->layers[nn->num_hidden_layers + 1].neurons[i].bias);
        for (int j = 0; j < nn->layers[nn->num_hidden_layers].num_neurons; j++) {  //weights from last hidden layer to output
            printf("%lf ", nn->layers[nn->num_hidden_layers + 1].neurons[i].weights[j]);
        }
        printf("]\n");
    }
}


//function to print results of each epoch
void printEpochResults(int epoch, int x, double *input, struct NeuralNetwork *nn, double *expected_output) {
    //print epoch and trainingset
    printf("Epoch: %d, T-Set: %d    Inputs: ", epoch + 1, x + 1);
    //print the inputs
    for (int k = 0; k < nn->num_inputs; k++) {
        printf("%g ", input[k]);
    }
    //print the calculated outputs
    printf("Outputs: ");
    for (int j = 0; j < nn->num_outputs; j++) {
        printf("%g ", nn->layers[nn->num_hidden_layers + 1].neurons[j].output);
    }
    //print the expected outputs
    printf("Expected Output: ");
    for (int j = 0; j < nn->num_outputs; j++) {
        printf("%g ", expected_output[j]);
    }
    printf("\n");   //nextline
}

//function to print the weights and biases
void printWeightsAndBiases(struct NeuralNetwork *nn) {
    for (int l = 1; l <= nn->num_hidden_layers + 1; l++) {
        struct Layer *layer = &nn->layers[l];
        printf("Layer %d:\n", l);

        for (int n = 0; n < layer->num_neurons; n++) {
            struct Neuron *neuron = &layer->neurons[n];
            printf("  Neuron %d - Bias: %f\n", n + 1, neuron->bias);

            //limit weights to the number of neurons in the previous layer
            printf("    Weights: ");
            for (int w = 0; w < nn->layers[l - 1].num_neurons; w++) {
                printf("%f ", neuron->weights[w]);
            }
            printf("\n");
        }
    }
}
//***********************************************************************************************

//in this format
//
// #inputs #hiddenlayers #outputs
// #neuronshiddenlayer1 #neuronshiddenlayer2 ...
// n1bias n1weight1 n1weight2 ....(depending on num inputs) n2bias n2weight1 n2weight2
// n1bias n1weight1 n1weight2 ....(depending on num neurons in hidden layer 1) n2bias n2weight1 n2weight2...
// ...
// #n1weight1...(last layer is output layer)

// Function to print the weights and network structure to a file (takes FILE* as argument)
void printWeightsToFile(struct NeuralNetwork *nn, FILE *file) {
    if (file == NULL) {
        return;
    }

    //print the first row with the number of inputs, hidden layers, and outputs
    fprintf(file, "%d %d %d\n", nn->num_inputs, nn->num_hidden_layers, nn->num_outputs);
    //print the sizes of the hidden layers in the second row
    for (int l = 1; l < nn->num_hidden_layers; l++) {
        fprintf(file, "%d ", nn->layers[l].num_neurons);
    }
    fprintf(file, "%d\n", nn->layers[nn->num_hidden_layers].num_neurons); //print the output layer size

    //go through each hidden layer and the output layer to print their neurons biases and weights
    for (int l = 1; l <= nn->num_hidden_layers+1; l++) {
        struct Layer *layer = &nn->layers[l];
        for (int n = 0; n < layer->num_neurons; n++) {
            struct Neuron *neuron = &layer->neurons[n];
            fprintf(file, " %.6f", neuron->bias);  //print the bias first
            for (int w = 0; w < nn->layers[l - 1].num_neurons; w++) {
                fprintf(file, " %.6f", neuron->weights[w]);  //print weights to 6 decimal places
            }
        }
        fprintf(file, "\n");
    }
}

struct NeuralNetwork* readNetworkFromFile(FILE *file){
    struct NeuralNetwork* nn = (struct NeuralNetwork*)malloc(sizeof(struct NeuralNetwork));
    //fead the first line for the number of inputs, hidden layers, and outputs
    fscanf(file, "%d %d %d", &nn->num_inputs, &nn->num_hidden_layers, &nn->num_outputs);

    //read the number of neurons in each hidden layer
    nn->hidden_layer_sizes = (int*)malloc(sizeof(int) * nn->num_hidden_layers); 
    for (int i = 0; i < nn->num_hidden_layers; i++) {
        fscanf(file, "%d", &nn->hidden_layer_sizes[i]);
    }
    //initialize each layer
    nn->layers = (struct Layer*)malloc(sizeof(struct Layer) * (nn->num_hidden_layers + 2));  //+2 for input and output layers
    //initialize the input layer
    nn->layers[0].num_neurons = nn->num_inputs;
    nn->layers[0].neurons = (struct Neuron*)malloc(sizeof(struct Neuron) * nn->num_inputs); //allocate memory for input neurons
    //initialize each hidden layer and the output layer
    for (int l = 1; l <= nn->num_hidden_layers + 1; l++) {
        int num_neurons_in_previous_layer = (l == 1) ? nn->num_inputs : nn->hidden_layer_sizes[l - 2];  //previous layer's neuron count
        if (l <= nn->num_hidden_layers) {
            nn->layers[l].num_neurons = nn->hidden_layer_sizes[l - 1];
        } else {
            nn->layers[l].num_neurons = nn->num_outputs;
        }
        nn->layers[l].neurons = (struct Neuron*)malloc(sizeof(struct Neuron) * nn->layers[l].num_neurons);
        
        //read the biases and weights for all neurons in the current layer
        for (int n = 0; n < nn->layers[l].num_neurons; n++) {
            struct Neuron *neuron = &nn->layers[l].neurons[n];
            fscanf(file, "%lf", &neuron->bias);
            neuron->weights = (double*)malloc(sizeof(double) * num_neurons_in_previous_layer);
            for (int w = 0; w < num_neurons_in_previous_layer; w++) {
                fscanf(file, "%lf", &neuron->weights[w]);
            }
        }
    }
    return nn;
}

//function to use the neural network
void useNeuralNetwork(struct NeuralNetwork *nn) {
    //ask for the activation function before the loop
    char activation_func[20];
    printf("Enter the activation function used for training (sigmoid, relu, etc): ");
    scanf("%s", activation_func);
    double *current_input = (double*)malloc(sizeof(double) * nn->num_inputs);

    while (1) {
        //get the inputs
        printf("Enter %d input values (separated by spaces):\n", nn->num_inputs);
        for (int i = 0; i < nn->num_inputs; i++) {
            scanf("%lf", &current_input[i]);
        }
        //use forward prop
        forwardPropagation(nn, current_input, activation_func);
        //output the result
        printf("Network outputs:\n");
        struct Layer *output_layer = &nn->layers[nn->num_hidden_layers + 1];
        for (int i = 0; i < output_layer->num_neurons; i++) {
            printf("Output neuron %d: %lf\n", i + 1, output_layer->neurons[i].output);
        }

        //ask if the user wants to continue or exit
        int continue_program;
        printf("\n");
        printf("Enter 0 to continue: ");
        scanf("%d", &continue_program);
        printf("\n");
        
        if (continue_program != 0) {
            printf("Done\n");

        }
    }
    free(current_input);
}

#endif