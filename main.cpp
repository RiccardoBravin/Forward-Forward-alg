#include "network.hpp"
#include "utility.hpp"
#include <random>
#include <time.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>


//read file composed as label, feature1, feature2, ... , featureN \n label, feature1, feature2, ... , featureN \n ...
//and save to vector of vectors row by row
std::vector<std::vector<float>> readFile (std::string file);

int vec_to_lab(std::vector<float> &vec, int classes);

int main(int argc, char const *argv[])
{
    srand(time(NULL));
    int labels = 10;
    std::cout << "Loading data..." << std::endl;
    //read data
    std::vector<std::vector<float>> train = readFile("MNIST_train.txt"); 
    std::vector<std::vector<float>> test  = readFile("MNIST_test.txt");
    std::cout << "\tFiles loaded" << std::endl;
    std::cout << "train size: " << train.size() << std::endl;
    std::cout << "test size: " << test.size() << std::endl;


    std::cout << "Generating auxiliary data..." << std::endl;
    //generate positive data for training by changing the label encoding
    std::vector<std::vector<float>> data_pos(train); //copy train data
    for(int i = 0; i < train.size(); i++){
        int label = data_pos[i][0]; //read label from current sample

        for(int j = 0; j < labels; j++){ //set all labels to 0
            data_pos[i][j] = 0;
        }

        data_pos[i][label] = 1; //set the correct label position to 1
    }

    std::cout << "\tPositive training data generated" << std::endl;

    //generate negative data for training by setting a random label to 1
    std::vector<std::vector<float>> data_neg(train); //copy train data
    
    for(int i = 0; i < train.size(); i++){
        int label = train[i][0];

        for(int j = 0; j < labels; j++){ //set all labels to 0
            data_neg[i][j] = 0;
        }

        int rand_label = rand()%labels-1;
        if(rand_label >= label) rand_label++; //avoid setting the same label
        
        data_neg[i][rand_label] = 1;
    }
    //shuffle negative data
    std::random_shuffle(data_neg.begin(), data_neg.end());

    std::cout << "\tNegative training data generated" << std::endl;

    //generate positive data for testing by changing the label encoding
    for(int i = 0; i < test.size(); i++){
        int label = test[i][0]; //read label from current sample

        for(int j = 0; j < labels; j++){ //set all labels to 0
            test[i][j] = 0;
        }

        test[i][label] = 1; //set the correct label position to 1
    }

    std::cout << "\tPositive test data generated" << std::endl;

    //define network
    int input_size = data_pos[0].size();
    int n_layers = 2;
    int epochs = 1;
    float lr = 0.01;
    int layers[n_layers+1] = {input_size, 200, 200};

    Network net(layers, n_layers, labels, lr);

    //test of forward pass
    std::cout << "Forward pass: " << std::endl;

    int guess = net.predict(test[0]);
    std::cout << "NN predicted: " << guess << std::endl;
    
    
    //training loop
    tic();
    std::cout << "train: " << std::endl;
    for(int i = 0; i < epochs; i++)
    {   
        std::cout << "epoch: " << i << std::endl;
        net.train(data_pos, data_neg);
        if(i > epochs/2){
            net.set_lr(lr/2);
        }
    }
    std::cout << "training complete: " << std::endl;
    toc();

    std::cout << "testing network: " << std::endl;
    int guesses[test.size()];
    #pragma omp parallel for
    for(int i = 0; i < test.size(); i++)
    {
        guesses[i] = net.predict(test[i]);
    }

    int count = 0;
    for(int i = 0; i < test.size(); i++)
    {
        if(vec_to_lab(test[i], labels) == guesses[i]) count++;
    }
    std::cout << "accuracy: " << (float)count/test.size() << std::endl;
    
    //net.print_net();
    return 0;
}

std::vector<std::vector<float>> readFile (std::string file){
std::ifstream data(file);

    std::vector<std::vector<float>> data_vec;
    if (!data.is_open()) {
        std::cerr << "Error opening file: " << file << std::endl;
        return data_vec; // Return an empty vector if the file cannot be opened
    }

    std::string line;
    std::string cell;
    std::vector<float> temp_vec;
    int count = 0;
    
    while(!data.eof()){
        
        std::getline(data, line);    
        std::stringstream lineStream(line);
        
        //insert label
        std::getline(lineStream, cell, ',');
        temp_vec.push_back(std::stof(cell));
        
        //insert scaled features
        while(std::getline(lineStream, cell, ',')){        
            temp_vec.push_back(std::stof(cell)/255);
        }      
    
        data_vec.push_back(temp_vec);
        temp_vec.clear();            
    }

    data.close();

    return data_vec;
}

int vec_to_lab(std::vector<float> &vec, int classes){
    int label = 0;
    for(int i = 0; i < classes; i++) {
        if(vec[i] == 1) {
            label = i;
            break;
        }
    }
    return label;
}