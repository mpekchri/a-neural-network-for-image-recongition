/*
 ============================================================================
 Name        : ergasia4.cu
 Author      : christopher
 Version     :
 Copyright   : @ copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include "Network.h"
#include <unistd.h>

__global__ void myKernel(float* a, float* b, float* c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void arrays_multiplication(int rows,int cols,double* matrix,double* vector,double* result) {

}

using namespace std;
void train(double learning_rate, Network* net);
int percentage_count(int* percentage, int start, int end);
void backpropagate(double* d_L);

int main(void) {

	int numOfLayers = 3;
	int* hsizes = new int[3];
	hsizes[0] = 784;
	hsizes[1] = 30;
	hsizes[2] = 10;

	Network* net = new Network(numOfLayers, hsizes);
	delete[] hsizes;

	// Cuda variables
	double *cuda_w, *cuda_d, *cuda_simg_der;
	int* sizes = new int[3];
	int* cuda_sizes;

	//net->train(2.3);
	train(2.3, net);

	/*
	 float *a,*b,*c;
	 float *d_a,*d_b,*d_c;
	 a = new float[100];
	 b = new float[100];

	 for(int i=0; i<100; i++){
	 a[i] = i*2;
	 b[i] = (i-1)*(i/2);
	 }
	 c = new float[100];

	 cudaMalloc((void**)&d_a,sizeof(float)*100);
	 cudaMalloc((void**)&d_b,sizeof(float)*100);
	 cudaMalloc((void**)&d_c,sizeof(float)*100);

	 cudaMemcpy(d_a,a,sizeof(int)*100,cudaMemcpyHostToDevice);
	 cudaMemcpy(d_b,b,sizeof(int)*100,cudaMemcpyHostToDevice);

	 myKernel<<<100,1>>>(d_a,d_b,d_c);
	 // cudaDeviceSynchronize();

	 // CPU blocks until memory is copied, memory copy starts only after kernel finishes
	 cudaMemcpy(c,d_c,sizeof(float)*100,cudaMemcpyDeviceToHost);
	 std::cout<<"fromGpu = "<<c[21]<<std::endl;

	 // clean up
	 cudaFree(d_a);
	 cudaFree(d_b);
	 cudaFree(d_c);
	 */

	cout << "SUCCESS -- ergasia4.cu file";
	return 0;
}

void init_cuda(int rows, int cols, double** w, double* d_L, int* sizes,
		double* sigm_derivative, double* cuda_w, double* cuda_d,
		double* cuda_sigm_der, int* cuda_sizes) {
	int w_size = rows * cols;
	int d_size = cols;
	int sigm_size = rows;
	sizes[0] = w_size;
	sizes[1] = d_size;
	sizes[2] = sigm_size;

	// allocate memory
	cudaMalloc((void**) &cuda_w, sizeof(double) * w_size);
	cudaMalloc((void**) &cuda_d, sizeof(double) * d_size);
	cudaMalloc((void**) &cuda_sigm_der, sizeof(double) * sigm_size);
	cudaMalloc((void**) &cuda_sizes, sizeof(int) * 3);

	// copy values from host in to device
	// TO-DO : use an ASYNCHRONOUS version of cudaMemcpy !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// first the ones that need no transpose
	cudaMemcpy(cuda_d, d_L, sizeof(double) * d_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_sigm_der, sigm_derivative, sizeof(double) * sigm_size,cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_sizes, sizes, sizeof(int) * 3, cudaMemcpyHostToDevice);

	// transpose w into an 1-D vector and copy it
	double* temp_w = new double[w_size];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			temp_w[i * cols + j] = w[i][j];
		}
	}
	cudaMemcpy(cuda_w, temp_w, sizeof(double) * w_size, cudaMemcpyHostToDevice);
	// wait until copy occurs (if you got an asynchronous version - not this one)and then delete temp_w
	cudaDeviceSynchronize();
	delete[] temp_w;

	// done
}

void reallocate_cuda(int rows, int cols, double* cuda_w, double* cuda_d,
		double* cuda_sigm_der, int* cuda_sizes) {
	// clean up
	cudaFree(cuda_w);
	cudaFree(cuda_d);
	cudaFree(cuda_sigm_der);

	int w_size = rows * cols;
	int d_size = cols;
	int sigm_size = rows;

	// reallocate
	cudaMalloc((void**) &cuda_w, sizeof(double) * w_size);
	cudaMalloc((void**) &cuda_d, sizeof(double) * d_size);
	cudaMalloc((void**) &cuda_sigm_der, sizeof(double) * sigm_size);
	cudaMalloc((void**) &cuda_sizes, sizeof(int) * 3);

	// done
}

void receive_cuda(int rows, int cols) {
	int w_size = rows * cols;
	int d_size = cols;
	int sigm_size = rows;


}

void send_cuda(int rows, int cols, int* sizes, double** w, double* d_L,
		double* sigm_derivative, double* cuda_w, double* cuda_d,
		double* cuda_sigm_der, double* cuda_sizes) {
	// reallocate _cuda(...) must have already been called
	int w_size = rows * cols;
	int d_size = cols;
	int sigm_size = rows;

	sizes[0] = w_size;
	sizes[1] = d_size;
	sizes[2] = sigm_size;

	// send data
	cudaMemcpy(cuda_d, d_L, sizeof(double) * d_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_sigm_der, sigm_derivative, sizeof(double) * sigm_size,
			cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_sizes, sizes, sizeof(int) * 3, cudaMemcpyHostToDevice);

	// transpose w into an 1-D vector and copy it
	double* temp_w = new double[w_size];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			temp_w[i * cols + j] = w[i][j];
		}
	}
	cudaMemcpy(cuda_w, temp_w, sizeof(double) * w_size, cudaMemcpyHostToDevice);
	// wait until copy occurs (if you got an asynchronous version - not this one)and then delete temp_w
	cudaDeviceSynchronize();
	delete[] temp_w;

	// done
}

void backpropagate(double* d_L) {

}

int percentage_count(int* percentage, int start, int end) {
	int result = 0;
	for (int i = start; i < end; i++) {
		result += percentage[i];
	}
	return result;
}

void train(double learning_rate, Network* net) {
	int epochs = 6000; // number of different inputs that will be used to train our network
	int batch_size = 1;
	int offset = 0;
	double *a, *y, *d_L, *cost;
	int y_int = 0;
	int* percentage = new int[epochs];
	for (int ep = offset; ep < offset + epochs; ep += (batch_size)) {
		for (int b = 0; b < batch_size; b++) {
			//a = read_tuple(ep + b, &y_int);
			a = net->read_tuple(ep + b, &y_int);
			net->feedforward(&a);
			y = net->transformOutput(y_int);
			cost = net->cost_derivative(
					net->getAlfa(net->getLayersNumber() - 1), y);
			d_L = net->hadamard_product(
					net->getLayersSize(net->getLayersNumber() - 1), cost,
					net->getSigmDerivative(net->getLayersNumber() - 1));
			net->backpropagate(d_L);
			net->update_sums();
			percentage[ep + b] = (int) net->getError(
					net->getAlfa(net->getLayersNumber() - 1), y_int);
			//debug();
			delete[] cost;
			delete[] y;
			for (int i = 0; i < net->getLayersNumber(); i++) {
				delete[] (net->getAlfa(i));
				delete[] (net->getSigmDerivative(i));
				delete[] (net->getDelta(i));
			}
		}
		net->gradient_descent(learning_rate, batch_size);
		net->reset_sums();
	}
	int p0 = percentage_count(percentage, 0, epochs / 4);
	cout << "number of training data used :                          " << epochs
			<< "\n";
	cout << "learning rate used :                                    "
			<< learning_rate << "\n";
	cout << "batch size used :                                       "
			<< batch_size << "\n";
	cout << "percentage of correct anwsers in 0 until 25 % is        "
			<< (1 - (double) p0 / (double) epochs) * 25 << " % \n";
	p0 = percentage_count(percentage, epochs / 4, epochs / 2);
	cout << "percentage of correct anwsers in 25 until 50 % is       "
			<< (1 - (double) p0 / (double) epochs) * 25 << " % \n";
	p0 = percentage_count(percentage, epochs / 2, epochs / (100 / 75));
	cout << "percentage of correct anwsers in 50 until 75 % is       "
			<< (1 - (double) p0 / (double) epochs) * 25 << " % \n";
	p0 = percentage_count(percentage, epochs / (100 / 75), epochs);
	cout << "percentage of correct anwsers in 75 until the end is    "
			<< (1 - (double) p0 / (double) epochs) * 25 << " % \n";
	delete[] percentage;
}
