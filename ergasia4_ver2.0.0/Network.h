/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Network.h
 * Author: christopher
 *
 * Created on August 2, 2017, 8:41 PM
 */

#ifndef NETWORK_H
#define NETWORK_H

namespace std {

class Network {
private:
    double **w,**b;
    int numOfLayers;
    int* sizeOfLayers;
public:
    Network();
    Network(int numOfLayers,int* sizes);
    Network(const Network& orig);
    virtual ~Network();
    void tries();
    double getRandom(int min,int max);
    double sigmoid(double z);
    void feedforward(double** a);
    double* read_tuple(int offset,int* y);
    int getOutput(double* out);
    
private:

};


}
#endif /* NETWORK_H */

