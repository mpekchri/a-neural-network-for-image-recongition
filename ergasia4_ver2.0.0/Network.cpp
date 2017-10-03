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

double* hadamard_product(int size, double* a, double* b);
double vector_mult(int size, double* a, double* b);
double** vector_mult_specific(int a_s, int b_s, double* a, double* b);
void vector_add(int size, double* a, double* b);
void matrix_add(int rows, int cols, double** a, double** b);
double* mull_feedforward(int cols, int rows, double** matrix, double* vector);

Network::Network() {
}

Network::Network(int numOfLayers, int* sizes) {
    this->num_of_layers = numOfLayers;
    s = new int[numOfLayers];
    for (int i = 0; i < numOfLayers; i++) {
        s[i] = sizes[i];
    }
    // matrices declaration
    w = new double*[num_of_layers];
    c_w = new double*[num_of_layers];
    b = new double*[num_of_layers];
    c_b = new double*[num_of_layers];
    delta = new double*[num_of_layers];
    sigm_derivative = new double*[num_of_layers];
    alfa = new double*[num_of_layers];

    alfa[0] = new double[s[0]];
    w[0] = NULL;
    b[0] = NULL;
    c_w[0] = NULL;
    c_b[0] = NULL;
    sigm_derivative[0] = NULL;
    delta[0] = NULL;
    for (int i = 1; i < numOfLayers; i++) {
        w[i] = new double[s[i - 1] * s[i]];
        c_w[i] = new double[s[i - 1] * s[i]];
        sigm_derivative[i] = new double[s[i]];
        b[i] = new double[s[i]];
        c_b[i] = new double[s[i]];
        delta[i] = new double[s[i]];
        alfa[i] = new double[s[i]];
    }
    initiallization();

}

Network::Network(const Network& orig) {

}

Network::~Network() {
    // clean up memory

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
        cost = cost_derivative(alfa[num_of_layers - 1], y);
        d_L = hadamard_product(s[num_of_layers - 1], cost, sigm_derivative[(num_of_layers - 1) - 1]);

        delete[] cost;
        delete[] a;
    }

}

double Network::sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

void Network::sigmoid(double** z, int size) {
    for (int i = 0; i < size; i++) {
        (*z)[i] = sigmoid((*z)[i]);
    }
}

void vector_add(int size, double* a, double* b) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
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

double* mull_feedforward(int rows, int cols, double* matrix, double* vector) {
    // TESTED
    // returns "cols x 1" vector
    double* temp = NULL;
    double* res = new double[cols];
    for (int j = 0; j < cols; j++) {
        temp = hadamard_product(rows, &matrix[j * rows], vector);
        res[j] = 0;
        for (int i = 0; i < rows; i++) {
            res[j] += temp[i];
        }
        delete[] temp;
    }
    return res;
}

double* mull_backpropagate(int rows, int cols, double* matrix, double* vector) {
    // TESTED
    // returns "rows x 1" vector
    double* temp = NULL;
    double* res = new double[rows];
    for (int j = 0; j < rows; j++) {
        temp = hadamard_product(cols, &matrix[j * cols], vector);
        res[j] = 0;
        for (int i = 0; i < cols; i++) {
            res[j] += temp[i];
        }
        delete[] temp;
    }

    return res;
}

double* compute_z(double* a, double* w, double* b, int rows, int cols) {
    double* result = mull_feedforward(rows, cols, w, a);
    vector_add(cols, result, b);
    return result;
}

void compute_sigm_der(double* a, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i]*(1 - a[i]);
    }
}

void Network::feedforward(double** a) {
    alfa[0] = a[0];
    for (int i = 1; i < num_of_layers; i++) {
        alfa[i] = compute_z(alfa[i - 1], w[i], b[i], s[i - 1], s[i]);
        sigmoid(&alfa[i], s[i]);
        compute_sigm_der(alfa[i], sigm_derivative[i], s[i]);
    }
    /*
    for (int x = 0; x < 10; x++) {
        cout << b[1][x] << " ";
    }
    cout << "\n";
    */
}

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

double* Network::transformOutput(int output) {
    // transforms a singleton input (named output:int) into 
    // a vector (named result:*double)
    double* result = new double[this->s[num_of_layers - 1]];
    for (int i = 0; i < s[num_of_layers - 1]; i++) {
        result[i] = 0;
    }
    result[output] = 1;
    return result;
}

double* Network::cost_derivative(double* a, double* y) {
    // derivative of C with respect to a (a == output layer's content   )
    double* result = new double[s[num_of_layers - 1]];
    for (int i = 0; i < s[num_of_layers - 1]; i++) {
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
    int batch_size = 100;
    int offset = 0;
    double *a, *y, *d_L, *cost;
    int y_int = 0;    
    for (int ep = offset; ep < offset + epochs; ep += (batch_size)) {
        reset_sums();
        for (int b = 0; b < batch_size; b++) {
            a = read_tuple(ep+b, &y_int);
            feedforward(&a);
            y = transformOutput(y_int);
            cost = cost_derivative(alfa[num_of_layers - 1], y);
            d_L = hadamard_product(s[num_of_layers - 1], cost, sigm_derivative[num_of_layers - 1]);
            backpropagate(d_L);
            update_sums();
            debug();
        }
        gradient_descent(learning_rate, batch_size);
    }
    // check
    int check_size = 100;
    int percentage = 0;
    for(int i=epochs; i<epochs+check_size; i++){
        a = read_tuple(i, &y_int);
        feedforward(&a);
        y = transformOutput(y_int);
        percentage += getError(alfa[num_of_layers - 1],y, y_int);
    }
    cout << "percentage of correct anwsers is " << (1-percentage/(double)check_size)*100 << " % \n";
}

void Network::gradient_descent(double learning_rate, int batch_size) {
    for (int i = 1; i < num_of_layers; i++) {
        for (int j = 0; j < s[i] * s[i - 1]; j++) {
            w[i][j] = w[i][j] + learning_rate * ((double) (1 / (double) batch_size)) * c_w[i][j];
        }
    }
    for (int i = 1; i < num_of_layers; i++) {
        for (int j = 0; j < s[i]; j++) {
            b[i][j] = b[i][j] + learning_rate * ((double) (1 / (double) batch_size)) * c_b[i][j];
        }
    }
}

void Network::backpropagate(double* d_L) {
    delta[num_of_layers - 1] = d_L;
    double* w_d;
    for (int i = num_of_layers - 2; i > 0; i--) {
        w_d = mull_backpropagate(s[i], s[i + 1], w[i + 1], delta[i + 1]);
        delta[i] = hadamard_product(s[i], w_d, sigm_derivative[i]);
        delete[] w_d;
    }

}

double* vector_mult(double* a, double* b, int rows, int cols) {
    // TESTED
    double* result = new double[rows * cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i * cols + j] = a[i] * b[j];
        }
    }
    return result;
}

void Network::compute_cost_w_derivative() {
    double* temp;
    for (int i = 1; i < num_of_layers; i++) {
        temp = vector_mult(alfa[i - 1], delta[i], s[i - 1], s[i]);
       
        vector_add(s[i - 1] * s[i], c_w[i], temp);
        delete[] temp;
    }
}

void Network::compute_cost_b_derivative() {
    for (int i = 1; i < num_of_layers; i++) {
        vector_add(s[i], c_b[i], delta[i]);
    }
}

void Network::update_sums() {
    compute_cost_w_derivative();
    compute_cost_b_derivative();
}

void Network::reset_sums() {
    for (int i = 1; i < num_of_layers; i++) {
        for (int j = 0; j < s[i] * s[i - 1]; j++) {
            c_w[i][j] = 0;
        }
    }
    for (int i = 1; i < num_of_layers; i++) {
        for (int j = 0; j < s[i]; j++) {
            c_b[i][j] = 0;
        }
    }
}

void Network::debug() {


}

int Network::getError(double* y_est, double* y,int y_val) {
    // returns 1 if it was error
    // 0 if it was correct
    int pos = 0;
    double min = pow(y[0]-y_est[0],2);
    for(int i=1; i<10; i++){
        //cout << pow(y[i]-y_est[i],2) <<" \n";
        if(pow(y[i]-y_est[i],2)<min){
            pos = i;
            min = pow(y[i]-y_est[i],2);
        }
    }
    cout << y_val << " " << pos << "\n";
    if (y_val == pos) {
        return 0;
    } else {
        return 1;
    }
}

double* Network::read_tuple(int offset, int* y) {
    double* res = new double[(s[0])];
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
        res[i] = (double) (((int) (temp[i])) / (double) 255.0);
        //res[i] = (double) (((int) (temp[i])) / (double) 1);
        //cout << res[i] << " ";
    }
    delete[] temp;
    file.close();

    // end

    return res;
}

void Network::initiallization() {
    for (int i = 1; i < num_of_layers; i++) {
        for (int j = 0; j < s[i]; j++) {
            b[i][j] = this->getRandom(-10, 10);
        }
    }
    for (int i = 1; i < num_of_layers; i++) {
        for (int j = 0; j < s[i - 1] * s[i]; j++) {
            w[i][j] = this->getRandom(-10, 10);
        }
    }
}

double Network::getRandom(int min, int max) {
    return (((max - min) * ((double) rand() / (double) RAND_MAX) + min)*100) / 100;
    // return floor(((max - min) * ((double) rand() / (double) RAND_MAX) + min)*100) / 100;
}



