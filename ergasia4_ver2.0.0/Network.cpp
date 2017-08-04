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

using namespace std;

Network::Network() {
}

Network::Network(int numOfLayers, int* sizes) {
    this->numOfLayers = numOfLayers;
    sizeOfLayers = new int[numOfLayers];
    for (int i = 0; i < numOfLayers; i++) {
        sizeOfLayers[i] = sizes[i];
    }
    w = new double*[numOfLayers];
    b = new double*[numOfLayers - 1];
    for (int i = 0; i < numOfLayers; i++) {
        w[i] = new double[sizeOfLayers[i]];
        for (int j = 0; j < sizeOfLayers[i]; j++) {
            w[i][j] = getRandom(0, 1);
        }
    }
    for (int i = 0; i < numOfLayers - 1; i++) {
        b[i] = new double[sizeOfLayers[i + 1]];
        for (int j = 0; j < sizeOfLayers[i + 1]; j++) {
            b[i][j] = getRandom(0, 1);
        }
    }

}

Network::Network(const Network& orig) {

}

Network::~Network() {
}

void Network::tries() {
    /*
    cout<<numOfLayers<<"\n";
    for(int i=0; i<numOfLayers; i++){
        cout<<sizeOfLayers[i]<<" ";
    }
     */

    double* a;
    int y=0;
    for (int i = 0; i < 1000; i++) {
        a = read_tuple(i,&y);
        feedforward(&a);
        int y_estimation = getOutput(a);
        // cout<<"y is "<<y<<"\n";
        // cout<<"y est is "<<y_estimation<<"\n";
        cout<<"error is "<<y-y_estimation<<"\n";
        delete[] a;
    }

}

double Network::sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

void Network::feedforward(double** a) {
    //double* res = new double[sizeOfLayers[numOfLayers-1]];
    for (int i = 0; i < numOfLayers - 1; i++) {
        double sum = 0;
        for (int x = 0; x < sizeOfLayers[i]; x++) {
            sum += (*a)[x] * w[i][x];
        }
        delete[] (*a);
        (*a) = new double[sizeOfLayers[i + 1]];
        for (int x = 0; x < sizeOfLayers[i + 1]; x++) {
            (*a)[x] = sigmoid(sum + b[i][x]);
        }
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

    double* Network::read_tuple(int offset,int* y) {
    double* res = new double[(sizeOfLayers[0])];
    unsigned char* temp;
    // returns a tuple (x,y)
    // y is integer and it can be taken as (int)res[0]
    // x is 784x1 array and it can be taken as &res[1]

    // return the label
    const char* path_sec = "/home/christopher/Documents/__KALOKAIRI17__/pitsianhs/data/train-labels.idx1-ubyte";
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
    //cout<<res[0]<<"\n";
    delete[] temp;
    file1.close();
    // read the image

    const char* path = "/home/christopher/Documents/__KALOKAIRI17__/pitsianhs/data/train-images.idx3-ubyte";
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
    imageSize = 784;
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

int Network::getOutput(double* out){
    int output = -1;
    double max = -1;
    for(int i=0; i<sizeOfLayers[numOfLayers-1]; i++){
        if(out[i]>max){
            max = out[i];
            output = i;
        }
    }
    return output;
}