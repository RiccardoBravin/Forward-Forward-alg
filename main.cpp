#include "network.hpp"
#include <random>
#include <time.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>


//read file composed as label, feature1, feature2, ... , featureN \n label, feature1, feature2, ... , featureN \n ...
//and save to vector of vectors
std::vector<std::vector<float>> readFile (std::string file)
{
    std::ifstream data(file);

    std::vector<std::vector<float>> data_vec;
    std::string line;
    std::string cell;
    std::vector<float> temp_vec;

    int count = 0;
    while(std::getline(data, line) && count < 100)
    {
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
        count++;
    }

    return data_vec;
}



int main(int argc, char const *argv[])
{
    srand(time(NULL));
    int labels = 10;

    //read data
    std::vector<std::vector<float>> data = readFile("MNIST_test.txt");
    
    // //print first line to check
    // for(int i = 0; i < data[0].size(); i++){
    //     std::cout << data[0][i] << " ";
    // }
    // std::cout << std::endl;
    
    //generate positive data
    std::vector<std::vector<float>> data_pos;
    for(int i = 0; i < data.size(); i++){
        std::vector<float> temp_vec;
        int label = data[i][0];

        for(int j = 0; j < 10; j++){
            temp_vec.push_back(0);
        }

        temp_vec[label] = 1;

        for(int j = 10; j < data[i].size(); j++){
            temp_vec.push_back(data[i][j]);
        }
        data_pos.push_back(temp_vec);
    }

    //generate negative data
    std::vector<std::vector<float>> data_neg;
    
    for(int i = 0; i < data.size(); i++){
        std::vector<float> temp_vec;
        int label = data[i][0];

        for(int j = 0; j < 10; j++){
            temp_vec.push_back(0);
        }
        temp_vec[rand()%10] = 1;
        for(int j = 10; j < data[i].size(); j++){
            temp_vec.push_back(data[i][j]);
        }
        data_neg.push_back(temp_vec);
    }


    //define network
    int input_size = data_pos[0].size();
    int n_layers = 2;
    int layers[n_layers+1] = {input_size, 500, 500};

    Network net(layers, n_layers, labels);

    //test forward pass
    std::cout << "forward pass: " << std::endl;
    
    std::cout << "input: " << std::endl;
    for(int i = 0; i < input_size; i++)
    {
        std::cout << data_pos[0][i] << " ";
    }
    std::cout << std::endl;

    int guess = net.predict(data_pos[0]);
    std::cout << "NN predicted: " << guess << std::endl;
    
    //test train
    std::cout << "train: " << std::endl;
    for(int i = 0; i < 100; i++)
    {   
        std::cout << "epoch: " << i << std::endl;
        net.train(data_pos, data_neg);
    }
    std::cout << "training complete: " << std::endl;

    //test output pos
    std::cout << "input: " << std::endl;
    for(int i = 0; i < input_size; i++)
    {
        std::cout << data_pos[0][i] << " ";
    }
    std::cout << std::endl;

    guess = net.predict(data_pos[0]);

    std::cout << "NN predicted: " << guess << std::endl;
    
    //test output neg
    std::cout << "input: " << std::endl;
    for(int i = 0; i < input_size; i++)
    {
        std::cout << data_neg[0][i] << " ";
    }
    std::cout << std::endl;

    guess = net.predict(data_neg[0]);

    std::cout << "NN predicted: " << guess << std::endl;

    return 0;
}
