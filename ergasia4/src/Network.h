/*
 * Network.h
 *
 *  Created on: Aug 9, 2017
 *      Author: chris
 */

#ifndef NETWORK_H_
#define NETWORK_H_

namespace std {


class Network {
private:
    double ***w;
    double **b,**sigm_derivative,**delta,**alfa;
    double ***w_sum,**b_sum;
    // cuda variables
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
    void sigmoid(double** z,int size);
    void feedforward(double** a);
    double* read_tuple(int offset,int* y);
    double* transformOutput(int output);
    double* sigmoid_derivative(double* sigmoid_result,int layers_id);
    double* cost_derivative(double* a,double* y);
    void backpropagate(double* d_L);
    void gradient_descent(double learning_rate,int batch_size);
    void update_sums();
    void reset_sums();
    double getError(double* y_est,int y);
    void debug();
    double* hadamard_product(int size, double* a, double* b);
    double* getAlfa(int i) {return alfa[i];}
    int getLayersNumber() {return numOfLayers;}
    int getLayersSize(int i) {return this->sizeOfLayers[i];}
    double* getDelta(int i) {return delta[i];}
    double* getSigmDerivative(int i) {return sigm_derivative[i];}
    void set_delta(int i,double* d){delta[i] = d;}
    double** getW(int i) {return w[i];}
    double* matrix_vector_mull(int cols, int rows, double** matrix, double* vector);
};



} /* namespace std */



#endif /* NETWORK_H_ */
