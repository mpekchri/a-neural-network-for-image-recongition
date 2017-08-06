/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Network.cpp
 * Author: christopher
 * 
 * Created on August 2, 2017, 8:41 PM
 */
#include "Network.h"
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <fstream>
#define num_of_pixels 784 

using namespace std;

Network::Network() {
}

Network::Network(int numOfLayers, int* sizes) {
    this->numOfLayers = numOfLayers;
    sizeOfLayers = new int[numOfLayers];
    for (int i = 0; i < numOfLayers; i++) {
        sizeOfLayers[i] = sizes[i];
    }
    w = new double*[numOfLayers - 1];
    b = new double*[numOfLayers - 1];
    sigm_derivative = new double*[numOfLayers - 1];
    delta = new double*[numOfLayers];
    alfa = new double*[numOfLayers];
    w_sum = new double*[numOfLayers - 1];
    b_sum = new double*[numOfLayers - 1];
    for (int i = 0; i < numOfLayers; i++) {
        w[i] = new double[sizeOfLayers[i + 1]];
        for (int j = 0; j < sizeOfLayers[i + 1]; j++) {
            w[i][j] = getRandom(0, 1);
        }
    }
    for (int i = 0; i < numOfLayers - 1; i++) {
        b[i] = new double[sizeOfLayers[i + 1]];
        for (int j = 0; j < sizeOfLayers[i + 1]; j++) {
            b[i][j] = getRandom(0, 1);
        }
    }
    for (int i = 0; i < numOfLayers - 1; i++) {
        sigm_derivative[i] = new double[sizeOfLayers[i + 1]];
    }
    for (int i = 0; i < numOfLayers; i++) {
        alfa[i] = new double[sizeOfLayers[i]];
    }
    
    for (int i = 0; i < numOfLayers - 1; i++) {
        w_sum[i] = new double[sizeOfLayers[i + 1]];
        for (int j = 0; j < sizeOfLayers[i + 1]; j++) {
            w_sum[i][j] = 0;
        }
    }
    for (int i = 0; i < numOfLayers - 1; i++) {
        b_sum[i] = new double[sizeOfLayers[i + 1]];        
        for (int j = 0; j < sizeOfLayers[i + 1]; j++) {
            b_sum[i][j] = 0;
        }
    }
    
}

Network::Network(const Network& orig) {

}

Network::~Network() {
}

double* hadamard_product(int size, double* a, double* b) {
    // returns the datamard product for vectors a and b 
    // (return a.*b in matlab)
    // size = length of arrays a and b
    double* result = new double[size];
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
    return result;
}

double* hadamard_product_singleton(int size, double a, double* b) {
    // returns the datamard product for vectors a and b 
    // (return a.*b in matlab)
    // size = length of arrays a and b
    double* result = new double[size];
    for (int i = 0; i < size; i++) {
        result[i] = a * b[i];
    }
    return result;
}

double vector_mult(int size, double* a, double* b) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void vector_add(int size, double* a, double* b) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

void Network::tries() {
    /*
    cout<<numOfLayers<<"\n";
    for(int i=0; i<numOfLayers; i++){
        cout<<sizeOfLayers[i]<<" ";
    }
     */

    double *a, *y, *d_L, *cost;
    int y_int = 0;
    for (int i = 0; i < 1000; i++) {
        a = read_tuple(i, &y_int);
        feedforward(&a);
        y = transformOutput(y_int);
        // d_L = hadamard_product(sizeOfLayers[numOfLayers-1],cost_derivative(a,y),sigmoid_derivative(a,(numOfLayers-1)));
        // or
        cost = cost_derivative(alfa[numOfLayers - 1], y);
        d_L = hadamard_product(sizeOfLayers[numOfLayers - 1], cost, sigm_derivative[(numOfLayers - 1) - 1]);
        cout << "new call\n";
        for (int i = 0; i < sizeOfLayers[numOfLayers - 1]; i++) {
            cout << "error is " << d_L[i] << "\n";
        }
        delete[] cost;
        delete[] a;
    }

}

double Network::sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

void Network::feedforward(double** a) {
    //double* res = new double[sizeOfLayers[numOfLayers-1]];
    alfa[0] = *a;
    for (int i = 0; i < numOfLayers - 1; i++) {
        double sum = 0;
        for (int x = 0; x < sizeOfLayers[i + 1]; x++) {
            sum += alfa[i][x] * w[i][x];
        }
        //delete[] (*a);
        //(*a) = new double[sizeOfLayers[i + 1]];
        for (int x = 0; x < sizeOfLayers[i + 1]; x++) {
            alfa[i + 1][x] = sigmoid(sum + b[i][x]);
        }
        sigm_derivative[i] = sigmoid_derivative(alfa[i + 1], i + 1);
    }
    // return (*a);
}

double Network::getRandom(int min, int max) {
    return floor(((max - min) * ((double) rand() / (double) RAND_MAX) + min)*100) / 100;
}

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

double* Network::read_tuple(int offset, int* y) {
    double* res = new double[(sizeOfLayers[0])];
    unsigned char* temp;
    // returns a tuple (x,y)
    // y is integer and it can be taken as (int)res[0]
    // x is 784x1 array and it can be taken as &res[1]

    // return the label
    const char* path_sec = "/home/chris/Documents/__KALOKAIRI17__/pitsianhs/data/train-labels.idx1-ubyte";
    ifstream file1(path_sec, ios::binary);
    int no_use = 0;
    file1.read((char *) &no_use, sizeof (no_use));
    no_use = reverseInt(no_use);
    if (no_use != 2049) {
        throw runtime_error("Invalid MNIST label file!");
    }
    file1.read((char *) &no_use, sizeof (no_use));
    no_use = reverseInt(no_use); // total num of labels
    temp = new unsigned char[1];
    for (int i = 0; i < offset; i++) {
        file1.read((char*) temp, 1);
    }
    file1.read((char*) temp, 1);
    y[0] = (double) temp[0];
    delete[] temp;
    file1.close();
    // read the image

    const char* path = "/home/chris/Documents/__KALOKAIRI17__/pitsianhs/data/train-images.idx3-ubyte";
    ifstream file(path, ios::binary);
    int imageSize = 0;
    int rows = 28, cols = 28;
    file.read((char *) &no_use, sizeof (no_use));
    no_use = reverseInt(no_use);
    if (no_use != 2051) {
        throw runtime_error("Invalid MNIST image file!");
    }
    file.read((char *) &no_use, sizeof (no_use)), no_use = reverseInt(no_use);
    file.read((char *) &rows, sizeof (rows)), rows = reverseInt(rows);
    file.read((char *) &cols, sizeof (cols)), cols = reverseInt(cols);
    //imageSize = rows*cols;
    imageSize = num_of_pixels;
    temp = new unsigned char[imageSize];
    for (int i = 0; i < offset; i++) {
        file.read((char *) temp, imageSize);
    }
    file.read((char *) temp, imageSize);
    for (int i = 0; i < imageSize; i++) {
        res[i] = (double) (((int) (temp[i])) / (double) 255);
        //cout << res[i] << " ";
    }
    delete[] temp;
    file.close();

    // end

    return res;
}

/*
int Network::getOutput(double* out) {
    int output = -1;
    double max = -1;
    for (int i = 0; i < sizeOfLayers[numOfLayers - 1]; i++) {
        if (out[i] > max) {
            max = out[i];
            output = i;
        }
    }
    return output;
}
 */

double* Network::transformOutput(int output) {
    // transforms a singleton input (named output:int) into 
    // a vector (named result:*double)
    double* result = new double[this->sizeOfLayers[numOfLayers - 1]];
    for (int i = 0; i < sizeOfLayers[numOfLayers - 1]; i++) {
        result[i] = 0;
    }
    result[output] = 1;
    return result;
}

double* Network::sigmoid_derivative(double* sigmoid_result, int layers_id) {
    // TO-DO : update documentation - works for every layer now

    // sigmoid_derivative = sigmoid(z)*(1-simgoid(z))
    // where z = a*w +b of the output layer
    // OPTIMAZATION :
    // in order to save computational time and recources ,since feedforward returns simgoid(z)
    // of the last layer,
    // we will compute : sigmoid_result(typeof *double) == feedforward result
    // and then sigmoid-derivative = sigmoid_derivative(sigmoid_result)
    double* result = new double[sizeOfLayers[layers_id]];
    for (int i = 0; i < sizeOfLayers[layers_id]; i++) {
        result[i] = sigmoid_result[i]*(1 - sigmoid_result[i]);
    }
    return result;
}

double* Network::cost_derivative(double* a, double* y) {
    // derivative of C with respect to a (a == output layer's content   )
    double* result = new double[sizeOfLayers[numOfLayers - 1]];
    for (int i = 0; i < sizeOfLayers[numOfLayers - 1]; i++) {
        result[i] = a[i] - y[i];
    }
    return result;
}

void Network::train() {
    int epochs = 40000; // number of different inputs that will be used to train our network
    int batch_size = 40;
    double *a, *y, *d_L, *cost;
    int y_int = 0;
    for (int ep = 0; ep < epochs; ep += batch_size) {
        for (int b = 0; b < batch_size; b++) {
            a = read_tuple(ep + b, &y_int);
            feedforward(&a);
            y = transformOutput(y_int);
            cost = cost_derivative(alfa[numOfLayers - 1], y);
            d_L = hadamard_product(sizeOfLayers[numOfLayers - 1], cost, sigm_derivative[(numOfLayers - 1) - 1]);
            backpropagate(d_L);
            cerr<<"ftanw edw ";
            update_sums();
            cerr<<"den ftanw edw ";
            delete[] cost;
            delete[] a;
        }
        gradient_descent();
        reset_sums();
    }
    
}

void Network::gradient_descent() {
    for (int i = 0; i < numOfLayers - 1; i++) {
        for (int j = 0; j < sizeOfLayers[i]; j++) {
            w[i][j] += w_sum[i][j];
        }
    }
    for (int i = 0; i < numOfLayers - 1; i++) {
        for (int j = 0; j < sizeOfLayers[i]; j++) {
            b[i][j] += b_sum[i][j];
        }
    }
}

void Network::update_sums() {
    for (int i = 1; i < numOfLayers; i++) {
        vector_add(sizeOfLayers[i - 1], w_sum[i], hadamard_product(sizeOfLayers[i - 1], delta[i], alfa[i - 1]));
        vector_add(sizeOfLayers[i - 1], b_sum[i], delta[i]);
    }
}

void Network::reset_sums() {
    for (int i = 0; i < numOfLayers - 1; i++) {
        w_sum[i] = new double[sizeOfLayers[i]];
        for (int j = 0; j < sizeOfLayers[i + 1]; j++) {
            w_sum[i][j] = 0;
        }
    }
    for (int i = 0; i < numOfLayers - 1; i++) {
        b_sum[i] = new double[sizeOfLayers[i + 1]];
        for (int j = 0; j < sizeOfLayers[i]; j++) {
            b_sum[i][j] = 0;
        }
    }
}

void Network::backpropagate(double* d_L) {
    delta[numOfLayers - 1] = d_L;
    for (int i = (numOfLayers - 1) - 1; i >= 0; i--) {
        // δx,l=((wl+1)Tδx,l+1)⊙σ′(zx,l)
        // remember : delta[i] is a *double pointer
        delta[i] = hadamard_product_singleton(sizeOfLayers[i + 1], vector_mult(sizeOfLayers[i + 1], w[i + 1], delta[i + 1]), sigm_derivative[i]);
    }

}

