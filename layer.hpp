#ifndef LAYER_HPP
#define LAYER_HPP

#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <memory>

#define THRESHOLD_REDUCTION_COEFFICIENT 1
#define TINY 1E-6

class Layer {
    public:
        Layer() = default;
        Layer(int in_features, int out_features, float lr = 0.01);
        ~Layer() = default;
        void forward(std::vector<float> &input, std::vector<float> &output);
        void forward(std::vector<std::vector<float>> &input, std::vector<std::vector<float>> &output);
        void train(std::vector<std::vector<float>> &x_pos, std::vector<std::vector<float>> &x_neg, std::vector<std::vector<float>> &out_pos, std::vector<std::vector<float>> &out_neg);
    
        int get_out_features() {return this->out_features;}

    private:
        int in_features;
        int out_features;
        std::vector<float> bias;
        std::vector<std::vector<float>> weights;
        int treshold;
        float lr;

        void L2Norm(const std::vector<float> &x, std::vector<float> &out);

};

Layer::Layer(int in_features, int out_features, float lr) {
    //std::cout << "layer real constructor" << std::endl;
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> d{0.0, 1.0};

    this->in_features = in_features;
    this->out_features = out_features;
    this->bias.resize(out_features);
    this->lr = lr;
    this->treshold = 2;//out_features * THRESHOLD_REDUCTION_COEFFICIENT;
    this->weights.resize(out_features, std::vector<float>(in_features));

    for (int i = 0; i < out_features; i++) {
        for (int j = 0; j < in_features; j++) {
            this->weights[i][j] = d(gen);
        }
        this->bias[i] = d(gen);
    }

}


void Layer::forward(std::vector<std::vector<float>> &input, std::vector<std::vector<float>> &output) {
    
    //iterate over all samples in the batch 
    #pragma omp parallel for
    for(int samp = 0; samp < input.size(); samp++){
        //normalizing the input vector with L2 (only direction matters)
        std::vector<float> norm(in_features); 
        L2Norm(input[samp], norm);
        //computing the output of the layer with ReLU activation
        //#pragma omp parallel for if(this->in_features > 1000)
        for (int i = 0; i < out_features; i++) {
            output[samp][i] = 0;
            for (int j = 0; j < in_features; j++) {
                output[samp][i] += norm[j] * this->weights[i][j];
            }
            output[samp][i] += this->bias[i];
            output[samp][i] = std::max(0.0f, output[samp][i]);
        }
    }
}

void Layer::forward(std::vector<float> &input, std::vector<float> &output) {
    //normalizing the input vector with L2 (only direction matters)
    std::vector<float> norm(in_features); 
    L2Norm(input, norm);
    //computing the output of the layer with ReLU activation
    //#pragma omp parallel for if(this->in_features > 1000)
    for (int i = 0; i < out_features; i++) {
        output[i] = 0;
        for (int j = 0; j < in_features; j++) {
            output[i] += norm[j] * this->weights[i][j];
        }
        output[i] += this->bias[i];
        output[i] = std::max(0.0f, output[i]);
    }   
}

void Layer::train(std::vector<std::vector<float>> &x_pos, std::vector<std::vector<float>> &x_neg, std::vector<std::vector<float>> &out_pos, std::vector<std::vector<float>> &out_neg){

    forward(x_pos, out_pos);
    forward(x_neg, out_neg);
    
    //the loss is (log(1/(1 + exp(-x + t))) + log(1/(1 + exp(y - t))))/2 but we directly compute the gradient
    for(int samp = 0; samp < out_pos.size(); samp++){
    
        float p_grad = 0;
        float n_grad = 0;
        
        //calculating squared avg of the output
        for(int j = 0; j < out_features; j++){
            p_grad += out_pos[samp][j] * out_pos[samp][j] / out_features;
            n_grad += out_neg[samp][j] * out_neg[samp][j] / out_features;
        }

        //std::cout << "squares pos: " << p_grad << std::endl;
        // std::cout << "squares neg: " << n_grad << std::endl;

        p_grad =  2 * this->lr * (1.0/(1 + std::exp(p_grad - this->treshold)) + TINY);
        n_grad =  -2 * this->lr * (1.0/(1 + std::exp(n_grad - this->treshold)) + TINY);

        // std::cout << "p_grad: " << p_grad << "\n";
        // std::cout << "n_grad: " << n_grad << "\n";

        #pragma omp parallel for //if(this->in_features > 1000)
        for(int i = 0; i < out_features; i++){
            for(int j = 0; j < in_features; j++){
                this->weights[i][j] += (p_grad * x_pos[samp][j] * out_pos[samp][i] + n_grad * x_neg[samp][j] * out_neg[samp][i]);
                //std::cout << "weight: " << (p_grad * x_pos[samp][j] * out_pos[samp][i] + n_grad * x_neg[samp][j] * out_neg[samp][i]) << "\n";
            }
            this->bias[i] += (p_grad * out_pos[samp][i] + n_grad * out_neg[samp][i]);
        }    
    }
}


void Layer::L2Norm(const std::vector<float> &x, std::vector<float> &out){
    float norm = 0;
    for (int i = 0; i < this->in_features; i++) {
        norm += x[i] * x[i];
    }
    norm = std::sqrt(norm);

    for (int i = 0; i < this->in_features; i++) {
        out[i] = x[i]/(norm + TINY);
    }

}



#endif // LAYER_HPP