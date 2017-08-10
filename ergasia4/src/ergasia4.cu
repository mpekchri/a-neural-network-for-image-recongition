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
#include <sys/time.h>

__device__ double* get_subvector(double* vector, int blockId,
		int threads_per_block) {
	return &vector[blockId * threads_per_block];
}

__device__ void hadamard_product_small(double* sh_a, double* sh_b,
		double* sh_result, int threads_per_block) {
	int thread_id = threadIdx.y * threads_per_block + threadIdx.x;
	// start the computations
	sh_a[thread_id] = sh_a[thread_id] * sh_b[thread_id];

	// store result form cache to global mem
	sh_result[thread_id] = sh_a[thread_id];
	//done
}

__device__ void array_sum_small(double* sha, double& result,
		int threads_per_warp, int threads_per_block, int numOfBlocks) {
	int block_id = blockIdx.y * numOfBlocks + blockIdx.x;
	int thread_id = threadIdx.y * threads_per_block + threadIdx.x;

	// start the computations
	for (int i = threads_per_warp; i < threads_per_block; i = i * 2) {
		// switch 1 : even warps add their's neighbors contents
		switch ((thread_id % i) % 2) {
		case 0:
			// thread_id  % i == even
			// add the "more next vector"
			sha[thread_id] = sha[thread_id] + sha[thread_id + threads_per_warp];
			break;
		default:
			// thread_id  % i == odd
			// do nothing
			break;
		}

		// switch2 : odd warps clean up their content
		switch ((thread_id % i) % 2) {
		case 0:
			// thread_id  % i == even
			// do nothing
			break;
		default:
			// thread_id  % i == odd
			// clean up
			sha[thread_id] = 0;
			break;
		}
	}
	// loop ended, sha[0:threads_per_warp] got the sum
	if (thread_id == 0) {
		// only the first thread will be used
		result = 0;
		for (int i = 0; i < threads_per_warp; i++) {
			(result) += sha[i];
		}
	}
	// sum is now stored in result's content
	// done
}

__global__ void array_mult(int rows, int cols, double* matrix, double* vector,
		double* result, int threads_per_block, int numOfBlocks,
		int blocks_per_maximumDim, int blocks_mod, int threads_per_warp) {
	extern __shared__ double shared[];	// sizeof(double)*256
	double* m = shared;
	double* v = (double*) &m[threads_per_block];

	// define id's
	int block_id = blockIdx.y * numOfBlocks + blockIdx.x;
	int thread_id = threadIdx.y * threads_per_block + threadIdx.x;
	int global_id = block_id + thread_id;
	double *w, *d, *res;

	// get data
	w = get_subvector(matrix, block_id, threads_per_block);
	d = get_subvector(vector,
			block_id
					- blocks_per_maximumDim
							* (block_id % blocks_per_maximumDim),
			threads_per_block);

	// fill chache
	if (block_id % blocks_per_maximumDim == 0) {
		// akriano block
		m[thread_id] = w[thread_id];
		m[thread_id] = m[thread_id] * (thread_id < blocks_mod); // if thread_id>=blocks_per_maximumDim then m[tread_id] = 0
		v[thread_id] = d[thread_id];
		v[thread_id] = v[thread_id] * (thread_id < blocks_mod); // if thread_id>=blocks_per_maximumDim then v[tread_id] = 0
		__syncthreads();
		// hadamard_product
		hadamard_product_small(m, v, m, threads_per_block);
	} else {
		// normal block
		m[thread_id] = w[thread_id];
		v[thread_id] = d[thread_id];
		__syncthreads();
		hadamard_product_small(m, v, m, threads_per_block);
	}

	// now w has the result
	// in order to retrieve the vector result[i] we compute sum(m[i][:]) where i = block_id

	// call array_sum or something like this
	array_sum_small(m, result[block_id], threads_per_warp, threads_per_block,
			numOfBlocks);

	// done
}

using namespace std;
void train(double learning_rate, Network* net, int* sizes, double* cuda_w,
		double* cuda_d, double* cuda_sigm_der, int* cuda_sizes,
		double* cuda_result, int threads_per_block, int numOfBlocks,
		int blocks_per_maximumDim, int blocks_mod, int threads_per_warp);
int percentage_count(int* percentage, int start, int end);
void backpropagate(double* d_L, Network* net, int* sizes, double* cuda_w,
		double* cuda_d, double* cuda_sigm_der, int* cuda_sizes,
		double* cuda_result, int threads_per_block, int numOfBlocks,
		int blocks_per_maximumDim, int blocks_mod, int threads_per_warp);

int findBlocks(int size, int threads_per_block);

int main(void) {

	int numOfLayers = 3;

	int* hsizes = new int[3];
	hsizes[0] = 784;
	hsizes[1] = 30;
	hsizes[2] = 10;

	Network* net = new Network(numOfLayers, hsizes);

	// Cuda variables
	int rows = hsizes[numOfLayers - 1];
	int cols = hsizes[numOfLayers - 2];
	int threads_per_block = 256;
	int threads_per_warp = 32;
	int numOfBlocks = hsizes[1] * findBlocks((rows * cols), threads_per_block);
	int blocks_per_maximumDim = findBlocks((rows * cols), threads_per_block);
	int blocks_mod = rows * cols
			- floor((rows * cols) / threads_per_block) * threads_per_block;
	double *cuda_w, *cuda_d, *cuda_sigm_der, *cuda_result;
	int* cuda_sizes;
	int* sizes = new int[7];
	sizes[0] = rows;
	sizes[1] = cols;
	sizes[2] = threads_per_block;
	sizes[3] = threads_per_warp;
	sizes[4] = numOfBlocks;
	sizes[5] = blocks_per_maximumDim;
	sizes[6] = blocks_mod;

	//net->train(2.3);
	train(2.3, net, sizes, cuda_w, cuda_d, cuda_sigm_der, cuda_sizes,
			cuda_result, threads_per_block, numOfBlocks, blocks_per_maximumDim,
			blocks_mod, threads_per_warp);

	cout << "SUCCESS -- ergasia4.cu file";
	delete[] hsizes;
	return 0;
}

int findBlocks(int size, int threads_per_block) {
	int res = (int) floor(size / threads_per_block);
	if (size / threads_per_block - res > 0) {
		res++;
	}
	return res;
}

void init_cuda(int rows, int cols, double** w, double* d_L, int* sizes,
		double* sigm_derivative, double* cuda_w, double* cuda_d,
		double* cuda_sigm_der, int* cuda_sizes, double* cuda_result) {
	int w_size = rows * cols;
	int d_size = cols;
	int sigm_size = rows;
	sizes[0] = rows;
	sizes[1] = cols;

	// allocate memory
	cudaMalloc((void**) &cuda_w, sizeof(double) * (w_size + sizes[6]));
	cudaMalloc((void**) &cuda_d, sizeof(double) * (d_size + sizes[6]));
	cudaMalloc((void**) &cuda_sigm_der, sizeof(double) * sigm_size);
	cudaMalloc((void**) &cuda_sizes, sizeof(int) * 7);
	cudaMalloc((void**) &cuda_result, sizeof(double) * rows);

	// copy values from host in to device
	// TO-DO : use an ASYNCHRONOUS version of cudaMemcpy !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// first the ones that need no transpose

	cudaMemcpy(cuda_sigm_der, sigm_derivative, sizeof(double) * sigm_size,
			cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_sizes, sizes, sizeof(int) * 3, cudaMemcpyHostToDevice);

	// transpose w into an 1-D vector and copy it
	double* dd = new double[d_size + sizes[6]];
	for (int i = 0; i < d_size; i++) {
		dd[i] = d_L[i];
	}
	for (int i = d_size; i < d_size + sizes[6]; i++) {
		dd[i] = 0;
	}
	cudaMemcpy(cuda_d, dd, sizeof(double) * (d_size + sizes[6]),
			cudaMemcpyHostToDevice);

	double* temp_w = new double[w_size + sizes[6]];

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			temp_w[i * cols + j] = w[i][j];
		}
	}
	// padarray temp_w in order no thread speacks to wrong memory
	for (int i = w_size; i < w_size + sizes[6]; i++) {
		temp_w[i] = 0;
	}

	cudaMemcpy(cuda_w, temp_w, sizeof(double) * (w_size + sizes[6]),
			cudaMemcpyHostToDevice);
	// wait until copy occurs (if you got an asynchronous version - not this one)and then delete temp_w
	cudaDeviceSynchronize();
	delete[] temp_w;
	delete[] dd;

	// done
}

void reallocate_cuda(int rows, int cols, double* cuda_w, double* cuda_d,
		double* cuda_sigm_der, int* cuda_sizes, int* sizes,
		double* cuda_result) {
	// clean up
	cudaFree(cuda_w);
	cudaFree(cuda_d);
	cudaFree(cuda_sigm_der);
	cudaFree(cuda_sizes);
	cudaFree(cuda_result);

	int w_size = rows * cols;
	int d_size = cols;
	int sigm_size = rows;

	// reallocate
	cudaMalloc((void**) &cuda_w, sizeof(double) * (w_size + sizes[6]));
	cudaMalloc((void**) &cuda_d, sizeof(double) * (d_size + sizes[6]));
	cudaMalloc((void**) &cuda_sigm_der, sizeof(double) * sigm_size);
	cudaMalloc((void**) &cuda_sizes, sizeof(int) * 7);
	cudaMalloc((void**) &cuda_result, sizeof(double) * rows);
	// done
}


void send_cuda(int rows, int cols, int* sizes, double** w, double* d_L,
		double* sigm_derivative, double* cuda_w, double* cuda_d,
		double* cuda_sigm_der, int* cuda_sizes) {
	// reallocate _cuda(...) must called first
	int w_size = rows * cols;
	int d_size = cols;
	int sigm_size = rows;

	// send data
	cudaMemcpy(cuda_sigm_der, sigm_derivative, sizeof(double) * sigm_size,
			cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_sizes, sizes, sizeof(int) * 7, cudaMemcpyHostToDevice);

	// transpose w into an 1-D vector and copy it
	double* dd = new double[d_size + sizes[6]];
	for (int i = 0; i < d_size; i++) {
		dd[i] = d_L[i];
	}
	for (int i = d_size; i < d_size + sizes[6]; i++) {
		dd[i] = 0;
	}
	cudaMemcpy(cuda_d, dd, sizeof(double) * (d_size + sizes[6]),
			cudaMemcpyHostToDevice);

	double* temp_w = new double[w_size + sizes[6]];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			temp_w[i * cols + j] = w[i][j];
		}
	}
	// padarray temp_w in order no thread speacks to wrong memory
	for (int i = w_size; i < w_size + sizes[6]; i++) {
		temp_w[i] = 0;
	}
	cudaMemcpy(cuda_w, temp_w, sizeof(double) * (w_size + sizes[6]),
			cudaMemcpyHostToDevice);
	// wait until copy occurs (if you got an asynchronous version - not this one)and then delete temp_w
	cudaDeviceSynchronize();
	delete[] temp_w;
	delete[] dd;

	// done
}

void backpropagate(double* d_L, Network* net, int* sizes, double* cuda_w,
		double* cuda_d, double* cuda_sigm_der, int* cuda_sizes,
		double* cuda_result, int threads_per_block, int numOfBlocks,
		int blocks_per_maximumDim, int blocks_mod, int threads_per_warp) {

	net->set_delta(net->getLayersNumber() - 1, d_L);

	init_cuda(sizes[0], sizes[1], net->getW(net->getLayersNumber() - 2), d_L,
			sizes, net->getSigmDerivative(net->getLayersNumber() - 1), cuda_w,
			cuda_d, cuda_sigm_der, cuda_sizes, cuda_result);

	double* w_d = new double[net->getLayersSize(sizes[1])];
	for (int i = net->getLayersNumber() - 2; i >= 0; i--) {
		// w_d = net->matrix_vector_mull(net->getLayersSize(i + 1),net->getLayersSize(i + 2), net->getW(i), net->getDelta(i + 1));
		// time it


		array_mult<<<numOfBlocks, threads_per_block, 2 * threads_per_block>>>(
				sizes[0], sizes[1], cuda_w, cuda_d, cuda_result,
				threads_per_block, numOfBlocks, blocks_per_maximumDim,
				blocks_mod, threads_per_warp);
		cudaDeviceSynchronize();
		cudaMemcpy(w_d, cuda_result, sizeof(float) * sizes[1],
				cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		/*
		w_d = net->matrix_vector_mull(net->getLayersSize(i + 1),net->getLayersSize(i + 2), net->getW(i), net->getDelta(i + 1));
		*/

		net->set_delta(i,
				net->hadamard_product(net->getLayersSize(i + 1), w_d,
						net->getSigmDerivative(i)));
		sizes[0] = net->getLayersSize(i + 1);
		sizes[1] = net->getLayersSize(i);
		reallocate_cuda(sizes[0], sizes[1], cuda_w, cuda_d, cuda_sigm_der,
				cuda_sizes, sizes, cuda_result);

		// free_cuda<<<...,...>>>();
		// reallocate_cuda<<<...,...>>>();
		// receive_cuda<<<...,...>>>();
		// send_cuda<<<...,...>>>();
		if (i > 0) {
			send_cuda(sizes[0], sizes[1], sizes,
					net->getW(net->getLayersNumber() - 2 - i),
					net->getDelta(i + 1), net->getSigmDerivative(i), cuda_w,
					cuda_d, cuda_sigm_der, cuda_sizes);
			cudaDeviceSynchronize();
		}

	}
	delete[] w_d;
	cudaFree(cuda_w);
	cudaFree(cuda_d);
	cudaFree(cuda_sigm_der);
	cudaFree(cuda_sizes);
	cudaFree(cuda_result);
	sizes[0] = net->getLayersSize(net->getLayersNumber() - 1);
	sizes[1] = net->getLayersSize(net->getLayersNumber() - 2);
	/*
	 delta[numOfLayers - 1] = d_L;
	 double* w_d;
	 for (int i = numOfLayers - 2; i >= 0; i--) {
	 // δx,l=((wl+1)Tδx,l+1)⊙σ′(zx,l)
	 w_d = matrix_vector_mull(sizeOfLayers[i + 1], sizeOfLayers[i + 2], w[i], delta[i + 1]);
	 delta[i] = hadamard_product(sizeOfLayers[i + 1], w_d, sigm_derivative[i]);
	 delete[] w_d;
	 }
	 */

}

int percentage_count(int* percentage, int start, int end) {
	int result = 0;
	for (int i = start; i < end; i++) {
		result += percentage[i];
	}
	return result;
}

void train(double learning_rate, Network* net, int* sizes, double* cuda_w,
		double* cuda_d, double* cuda_sigm_der, int* cuda_sizes,
		double* cuda_result, int threads_per_block, int numOfBlocks,
		int blocks_per_maximumDim, int blocks_mod, int threads_per_warp) {
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
			// start timer
			struct timeval t1, t2;
			gettimeofday(&t1, 0);
			//int times,timed;
			//times=clock();
			backpropagate(d_L, net, sizes, cuda_w, cuda_d, cuda_sigm_der,
					cuda_sizes, cuda_result, threads_per_block, numOfBlocks,
					blocks_per_maximumDim, blocks_mod, threads_per_warp);
			// stop the timer
			gettimeofday(&t2, 0);

			double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
			//timed=clock();
			//times=timed-times;
			cerr<<"Elasped time is "<<time<<" millisec \n";
			//net->backpropagate(d_L);
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
