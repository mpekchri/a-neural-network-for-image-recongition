/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: christopher
 *
 * Created on August 2, 2017, 8:39 PM
 */

#include <cstdlib>

#include "Network.h"
#include <iostream>
#include <unistd.h>

using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {
    
    int numOfLayers = 3;
    int* sizes = new int[3];
    sizes[0] = 784;
    sizes[1] = 30;
    sizes[2] = 10;
    
    Network* net = new Network(numOfLayers,sizes);
    delete[] sizes;
    
    net->train(8.5);
    
    return 0;
}
