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
    double **w;
    double **b,**sigm_derivative,**delta,**alfa;
    double **c_w,**c_b;
    int num_of_layers;
    int* s;     // size of layers
public:
    Network();
    Network(int numOfLayers,int* sizes);
    Network(const Network& orig);
    virtual ~Network();
    void initiallization();
    void tries();
    double getRandom(int min,int max);
    double sigmoid(double z);
    void sigmoid(double** z,int size);
    void feedforward(double** a);
    double* read_tuple(int offset,int* y);
    double* transformOutput(int output);
    double* cost_derivative(double* a,double* y);
    void backpropagate(double* d_L);
    void train(double learning_rate);
    void gradient_descent(double learning_rate,int batch_size);
    void update_sums();
    void reset_sums();
    int getError(double* y_est, double* y,int y_val);
    void debug();
    void compute_cost_w_derivative() ;
    void compute_cost_b_derivative() ;
};


}
#endif /* NETWORK_H */

