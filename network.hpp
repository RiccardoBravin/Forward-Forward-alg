#ifndef NETWORK_CPP
#define NETWORK_CPP

#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include "layer.hpp"


class Network{
    public:
        Network(int dims[], int n_layers, int labels, float lr = 0.01);
        int predict(std::vector<float> &input);
        void train(std::vector<std::vector<float>> &x_pos, std::vector<std::vector<float>> &x_neg);

        void print_net();
    private:
        int n_layers;
        std::vector<Layer> layers;
        float lr;
        int labels;
        int max_size;

};

Network::Network(int dims[], int n_layers, int labels, float lr) {
    this->n_layers = n_layers;
    this->labels = labels;
    this->lr = lr;
    this->max_size = 0;
    for (int i = 0; i < n_layers; i++) {
        this->layers.emplace_back(dims[i], dims[i+1], lr);
        this->max_size = std::max(dims[i+1], max_size);
    }
}


int Network::predict(std::vector<float> &input) {

    std::vector<float> temp_in(max_size);
    std::vector<float> temp_out(max_size);
    float results[labels];
    std::fill(results, results + labels, 0);

    //deriving label from input
    int label = 0;
    for(int i = 0; i < labels; i++) {
        if(input[i] == 1) {
            label = i;
            break;
        }
    }

    //run a forward pass for each label
    for(int run = 0; run < labels; run++) {
        //std::cout << "label: " << run << std::endl;
        for(int i = 0; i < labels; i++) {
            input[i] = 0;
        }
        input[run] = 1;

        //forward pass
        this->layers[0].forward(input, temp_in);

        // for (int j = 0; j < this->layers[0].get_out_features(); j++) {
        //     results[run] += temp_in[j] * temp_in[j] / this->layers[0].get_out_features();
        // }

        for (int i = 1; i < this->n_layers; i++) {    
            this->layers[i].forward(temp_in, temp_out);
            //calculate average of the square of output and add to results
            // for (int j = 0; j < this->layers[i].get_out_features(); j++) {
            //     results[run] += temp_out[j] * temp_out[j] / this->layers[i].get_out_features();
            // }
            
            std::swap(temp_in, temp_out);
            
        }

        for (int j = 0; j < this->layers[this->n_layers-1].get_out_features(); j++) {
            results[run] += temp_out[j] * temp_out[j] / this->layers[this->n_layers-1].get_out_features();
        }
            
        
        //std::cout << "results: " << results[run] << std::endl;

    }
    //resetting input
    input[labels-1] = 0;
    input[label] = 1;

    return std::distance(results, std::max_element(results, results + labels));
}

void Network::train(std::vector<std::vector<float>> &x_pos, std::vector<std::vector<float>> &x_neg) {

    std::vector<std::vector<float>> out_pos(x_pos.size(), std::vector<float>(max_size));
    std::vector<std::vector<float>> out_neg(x_pos.size(), std::vector<float>(max_size));
    std::vector<std::vector<float>> temp_pos(x_pos.size(), std::vector<float>(max_size));
    std::vector<std::vector<float>> temp_neg(x_pos.size(), std::vector<float>(max_size));


    this->layers[0].train(x_pos, x_neg, temp_pos, temp_neg);
    
    for(int i = 1; i < this->n_layers; i++) {
        this->layers[i].train(temp_pos, temp_neg, out_pos, out_neg);
        std::swap(temp_pos, out_pos);
        std::swap(temp_neg, out_neg);
    }

}


void Network::print_net(){
    for(int i = 0; i < this->n_layers; i++){
        std::cout << "layer: " << i << std::endl;
        this->layers[i].print_weights();
    }
}



#endif //NETWORK_CPP