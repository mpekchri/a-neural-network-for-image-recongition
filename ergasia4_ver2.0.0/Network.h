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
    double **w,**b,**sigm_derivative,**delta,**alfa;
    double **w_sum,**b_sum;
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
    // int getOutput(double* out);
    double* transformOutput(int output);
    double* sigmoid_derivative(double* sigmoid_result,int layers_id);
    double* cost_derivative(double* a,double* y);
    void backpropagate(double* d_L);
    void train();
    void gradient_descent();
    void update_sums();
    void reset_sums();
};


}
#endif /* NETWORK_H */

