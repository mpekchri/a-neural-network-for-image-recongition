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
#include <cmath>
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
    w = new double**[numOfLayers - 1];
    b = new double*[numOfLayers - 1];
    sigm_derivative = new double*[numOfLayers];
    alfa = new double*[numOfLayers];
    delta = new double*[numOfLayers];
    w_sum = new double**[numOfLayers - 1];
    b_sum = new double*[numOfLayers - 1];
    for (int i = 0; i < numOfLayers - 1; i++) {
        w[i] = new double*[sizeOfLayers[i + 1]];
        for (int j = 0; j < sizeOfLayers[i + 1]; j++) {
            w[i][j] = new double[sizeOfLayers[i]];
            for (int x = 0; x < sizeOfLayers[i]; x++) {
                // w[i][j][x] = getRandom(0, 1);
                w[i][j][x] = getRandom(-1, 1);
            }
        }
    }
    for (int i = 1; i < numOfLayers; i++) {
        b[i - 1] = new double[sizeOfLayers[i]];
        for (int j = 0; j < sizeOfLayers[i]; j++) {
            // b[i - 1][j] = getRandom(0, 1);
            b[i - 1][j] = getRandom(-1, 1);
        }
    }
    for (int i = 0; i < numOfLayers - 1; i++) {
        w_sum[i] = new double*[sizeOfLayers[i + 1]];
        for (int j = 0; j < sizeOfLayers[i + 1]; j++) {
            w_sum[i][j] = new double[sizeOfLayers[i]];
            for (int x = 0; x < sizeOfLayers[i]; x++) {
                w_sum[i][j][x] = 0.0;
            }
        }
    }
    for (int i = 1; i < numOfLayers; i++) {
        b_sum[i - 1] = new double[sizeOfLayers[i]];
        for (int j = 0; j < sizeOfLayers[i]; j++) {
            b_sum[i - 1][j] = 0.0;
        }
    }
}

Network::Network(const Network& orig) {

}

Network::~Network() {
    // clean up memory
    for (int i = 0; i < numOfLayers - 1; i++) {

        for (int j = 0; j < sizeOfLayers[i + 1]; j++) {
            delete[] w[i][j];
        }
        delete[] w[i];
    }
    delete[] w;
    for (int i = 1; i < numOfLayers; i++) {
        delete[] b[i - 1];
    }
    delete[] b;
    for (int i = 0; i < numOfLayers - 1; i++) {

        for (int j = 0; j < sizeOfLayers[i + 1]; j++) {
            delete[] w_sum[i][j];
        }
        delete[] w_sum[i];
    }
    delete[] w_sum;
    for (int i = 1; i < numOfLayers; i++) {
        delete[] b_sum[i - 1];
    }
    delete[] b_sum;
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

double vector_mult(int size, double* a, double* b) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

double** vector_mult_specific(int a_s, int b_s, double* a, double* b) {
    double** res = new double*[a_s];
    double temp;
    for (int i = 0; i < a_s; i++) {
        res[i] = new double[b_s];
        for (int j = 0; j < b_s; j++) {
            res[i][j] = a[i] * b[j];
        }
    }
    return res;
}

void vector_add(int size, double* a, double* b) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

void matrix_add(int rows, int cols, double** a, double** b) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            a[i][j] += b[i][j];
        }
    }
}

double* matrix_vector_mull(int cols, int rows, double** matrix, double* vector) {
    // TESTED 
    // returns "rows x 1" vector 
    double* result = new double[cols];
    for (int i = cols - 1; i >= 0; i--) {
        result[i] = 0.0;
    }
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            result[j] += matrix[i][j] * vector[i];
        }
    }
    return result;
}

void Network::tries() {
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

void Network::sigmoid(double** z, int size) {
    for (int i = 0; i < size; i++) {
        // (*z)[i] = 1.0 / (1.0 + exp(-(*z)[i]));
        (*z)[i] = sigmoid((*z)[i]);
    }
}

void Network::feedforward(double** a) {
    //double* vec_a_w;
    alfa[0] = a[0];
    double* sigm = new double[sizeOfLayers[0]];
    for (int i = 0; i < sizeOfLayers[0]; i++) {
        sigm[i] = alfa[0][i];
    }
    sigmoid(&sigm, sizeOfLayers[0]);
    sigm_derivative[0] = sigmoid_derivative(sigm, 0);
    delete[] sigm;
    for (int i = 0; i < numOfLayers - 1; i++) {
        alfa[i + 1] = matrix_vector_mull(sizeOfLayers[i], sizeOfLayers[i + 1], w[i], alfa[i]);
        vector_add(sizeOfLayers[i + 1], alfa[i + 1], b[i]); // result is stored in alfa[i+1]
        sigmoid(&alfa[i + 1], sizeOfLayers[i + 1]);
        sigm_derivative[i + 1] = sigmoid_derivative(alfa[i + 1], i + 1);
    }
    // return (*a);
}

double Network::getRandom(int min, int max) {
    return (((max - min) * ((double) rand() / (double) RAND_MAX) + min)*100) / 100;
    // return floor(((max - min) * ((double) rand() / (double) RAND_MAX) + min)*100) / 100;
}

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

double* Network::read_tuple(int offset, int* y) {   //int osffset now is the NUMBER of the requested item starting from 0
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
    file1.seekg(8+offset);
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
    file.seekg(16 + offset*num_of_pixels);
    file.read((char *) temp, imageSize);
    for (int i = 0; i < imageSize; i++) {
        res[i] = (double) (((int) (temp[i])) / (double) 255.0);
        //res[i] = (double) (((int) (temp[i])) / (double) 1);
        //cout << res[i] << " ";
    }
    delete[] temp;
    file.close();

    // end

    return res;
}

double* Network::transformOutput(int output) {
    // transforms a singleton input (named output:int) into 
    // a vector (named result:*double)
    double* result = new double[this->sizeOfLayers[numOfLayers - 1]];
    for (int i = 0; i < sizeOfLayers[numOfLayers - 1]; i++) {
        result[i] = 0.0;
    }
    result[output] = 1.0;
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

int percentage_count(int* percentage, int start, int end) {
    int result = 0;
    for (int i = start; i < end; i++) {
        result += percentage[i];
    }
    return result;
}

void Network::train(double learning_rate) {
    int epochs = 10000; // number of different inputs that will be used to train our network
    int batch_size = 1;
    int offset = 0;
    double *a, *y, *d_L, *cost;
    int y_int = 0;
    int* percentage = new int[epochs];
    for (int ep = offset; ep < offset + epochs; ep += (batch_size)) {
        //pososto = 0;
        for (int b = 0; b < batch_size; b++) {
            //a = read_tuple(ep + b, &y_int);
            a = read_tuple(ep + b, &y_int);
            feedforward(&a);
            y = transformOutput(y_int);
            cost = cost_derivative(alfa[numOfLayers - 1], y);
            d_L = hadamard_product(sizeOfLayers[numOfLayers - 1], cost, sigm_derivative[numOfLayers - 1]);
            backpropagate(d_L);
            update_sums();
            percentage[ep + b] = (int) getError(alfa[numOfLayers - 1], y_int);
            //debug();
            delete[] cost;
            delete[] y;
            for (int i = 0; i < numOfLayers; i++) {
                delete[] alfa[i];
                delete[] sigm_derivative[i];
                delete[] delta[i];
            }
        }
        // cerr << "epoch " << ep + batch_size << " with ratio " << (1 - (double) pososto / (double) batch_size)*100 << " % \n";
        gradient_descent(learning_rate, batch_size);
        reset_sums();
    }
    int p0 = percentage_count(percentage, 0, epochs / 4);
    cout << "number of training data used :                          " << epochs << "\n";
    cout << "learning rate used :                                    " << learning_rate << "\n";
    cout << "batch size used :                                       " << batch_size << "\n";
    cout << "percentage of correct anwsers in 0 until 25 % is        " << (1 - (double) p0 / (double) epochs)*25 << " % \n";
    p0 = percentage_count(percentage, epochs / 4, epochs / 2);
    cout << "percentage of correct anwsers in 25 until 50 % is       " << (1 - (double) p0 / (double) epochs)*25 << " % \n";
    p0 = percentage_count(percentage, epochs / 2, epochs / (100 / 75));
    cout << "percentage of correct anwsers in 50 until 75 % is       " << (1 - (double) p0 / (double) epochs)*25 << " % \n";
    p0 = percentage_count(percentage, epochs / (100 / 75), epochs);
    cout << "percentage of correct anwsers in 75 until the end is    " << (1 - (double) p0 / (double) epochs)*25 << " % \n";
    delete[] percentage;
}

void Network::gradient_descent(double learning_rate, int batch_size) {
    // update w and b according to w_sum and b_sum
    for (int i = 0; i < numOfLayers - 1; i++) {
        for (int j = 0; j < sizeOfLayers[i + 1]; j++) {
            for (int x = 0; x < sizeOfLayers[i]; x++) {
                w[i][j][x] -= learning_rate * (1 / batch_size) * w_sum[i][j][x];
            }
        }
        //cerr<<"w sum "<<w_sum[i][0][0]<<" -- ";
        //cerr<<"w "<<w[i][0][0]<<" \n";
    }
    for (int i = 1; i < numOfLayers; i++) {
        for (int j = 0; j < sizeOfLayers[i]; j++) {
            b[i - 1][j] -= learning_rate * (1 / batch_size) * b_sum[i - 1][j];
        }
    }
}

void Network::update_sums() {
    double** mul;
    for (int i = 0; i < numOfLayers - 1; i++) {
        mul = vector_mult_specific(sizeOfLayers[i + 1], sizeOfLayers[i], delta[i + 1], alfa[i]);
        matrix_add(sizeOfLayers[i + 1], sizeOfLayers[i], w_sum[i], mul);
        for (int k = 0; k < sizeOfLayers[i + 1]; k++) {
            //cerr<<sizeOfLayers[i+1]<<" "<<k<<"\n";
            delete[] mul[k];
        }
        delete[] mul;
    }
    for (int i = 0; i < numOfLayers - 1; i++) {
        vector_add(sizeOfLayers[i + 1], b_sum[i], delta[i + 1]);
    }
}

void Network::reset_sums() {
    for (int i = 0; i < numOfLayers - 1; i++) {
        for (int j = 0; j < sizeOfLayers[i + 1]; j++) {
            for (int x = 0; x < sizeOfLayers[i]; x++) {
                w_sum[i][j][x] = 0.0;
            }
        }
    }
    for (int i = 1; i < numOfLayers; i++) {
        for (int j = 0; j < sizeOfLayers[i]; j++) {
            b_sum[i - 1][j] = 0.0;
        }
    }
}

void Network::backpropagate(double* d_L) {
    delta[numOfLayers - 1] = d_L;
    double* w_d;
    for (int i = numOfLayers - 2; i >= 0; i--) {
        // δx,l=((wl+1)Tδx,l+1)⊙σ′(zx,l)
        w_d = matrix_vector_mull(sizeOfLayers[i], sizeOfLayers[i + 1], w[i], delta[i + 1]);
        delta[i] = hadamard_product(sizeOfLayers[i], w_d, sigm_derivative[i]);
        delete[] w_d;
    }

}

void Network::debug() {
    for (int i = 0; i < 10; i++) {
        cerr << "axx is " << alfa[2][i] << " \n";
    }
}

double Network::getError(double* y_est, int y) {
    // returns 1 if it was error
    // 0 if it was correct

    double max = -1;
    int pos = -1;
    for (int i = 0; i < 10; i++) {
        if (y_est[i] > max) {
            max = y_est[i];
            pos = i;
        }
    }

    if (y == pos) {
        return 0.0;
    } else {
        return 1.0;
    }
}
