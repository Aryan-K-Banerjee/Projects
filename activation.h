#ifndef ACTIVATION_H
#define ACTIVATION_H

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

//activation functions
double sigmoid(double x);
double dSigmoid(double x);

double activation(double x, char* func) {
    if (strcmp(func, "sigmoid") == 0) {
        return sigmoid(x);
    } else {
        printf("Activation function isn't included!\n");
        return 0.0; // Default return value for unsupported functions
    }
}

double dActivation(double x, char* func) {
    if (strcmp(func, "sigmoid") == 0) {
        return dSigmoid(x);
    } else {
        printf("Activation function derivative isn't included!\n");
        return 0.0; // Default return value for unsupported functions
    }
}

double sigmoid(double x){
    return 1 / (1+exp(-x));
}

double dSigmoid(double x){
    return x*(1-x);
}

#endif