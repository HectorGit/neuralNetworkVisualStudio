/*
==============================================================================

Main.cpp
Created: 29 Sep 2017 3:10:51pm
Author:  Owner

==============================================================================
*/
#include "stdafx.h"
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>
#include <iostream>
#include "NeuralNetwork.h"

int main() {

	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
	std::vector<int> paramSizes = { 3,5,3 };
	NeuralNetwork* aNet = new NeuralNetwork(paramSizes);

	/*Eigen::MatrixXf yTemp = Eigen::MatrixXf::Zero(3, 1); //10 by 1 just one number - a classification
	yTemp(0, 0) = 1;
	//yTemp(1, 0) = 0;
	//yTemp(2, 0) = 0;

	Eigen::MatrixXf yTemp2 = Eigen::MatrixXf::Zero(3, 1); //10 by 1 just one number - a classification
	yTemp2(1, 0) = 1;
	//yTemp2(1, 0) = 0;
	//yTemp2(2, 0) = 0;

	Eigen::MatrixXf yTemp3 = Eigen::MatrixXf::Zero(3, 1); //10 by 1 just one number - a classification
	//yTemp3(0, 0) = 0;
	//yTemp3(1, 0) = 0;
	yTemp3(2, 0) = 1;*/

	for (int i = 0; i < aNet->mini_batch_size * 5; i++) {

		Eigen::MatrixXf temp = Eigen::MatrixXf::Random(paramSizes[0], 1);
		aNet->all_Xs.emplace_back(temp); 
		/*if (i % 3 == 0) {
			aNet->all_Ys.emplace_back(yTemp);
		}
		else if(i%3 == 1){
			aNet->all_Ys.emplace_back(yTemp2);
		}
		else if(i%3 ==2) {
			aNet->all_Ys.emplace_back(yTemp3);
		}*/
		float maxVal = 0;
		Eigen::MatrixXi::Index highestNumIndex;
		maxVal = aNet->all_Xs[i].col(0).maxCoeff(&highestNumIndex);

		Eigen::MatrixXf yTrainingTemp = Eigen::MatrixXf::Zero(paramSizes[0], 1);
		std::cout << "\n highest num index: " << highestNumIndex << "\n";
		yTrainingTemp((int)highestNumIndex, 0) = 1;
		aNet->all_Ys.emplace_back(yTrainingTemp);
	
	}

	/*for (int i = 0; i < aNet->mini_batch_size; i++) {
		std::cout << "\n\n";
		std::cout << "all_Xs: \n" << aNet->all_Xs[i].format(CleanFmt);
		std::cout << "\n\n";
		std::cout << "all_Ys: \n" << aNet->all_Ys[i].format(CleanFmt);
		std::cout << "\n\n";
	}*/

	printf(" \n--------------TRAINING--------------\n");
	printf(" calling stochastic gradient descent\n");

	aNet->stochasticGradientDescent();
	
	printf(" \n--------------TESTING--------------\n");
	printf("          calling classify          \n");

	for (int i = 0; i < aNet->mini_batch_size; i++) {
		Eigen::MatrixXf testingTemp = Eigen::MatrixXf::Random(paramSizes[0], 1);
		aNet->dummyTestDataSet_Xs.emplace_back(testingTemp);		
		int maxVal = 0;
		Eigen::MatrixXi::Index highestNumIndex;
		maxVal = aNet-> dummyTestDataSet_Xs[i].col(0).maxCoeff(&highestNumIndex);
	
		Eigen::MatrixXf yTestingTemp = Eigen::MatrixXf::Zero(paramSizes[0], 1);
		yTestingTemp(highestNumIndex, 0) = 1;
		aNet->dummyTestDataSet_Ys.emplace_back(yTestingTemp);
	}
	
	/*for (int i = 0; i < aNet->mini_batch_size; i++) {
		std::cout << "\n\n";
		std::cout << "dummyTestDataSet_Xs: \n" << aNet->dummyTestDataSet_Xs[i].format(CleanFmt);
		std::cout << "\n\n";
		std::cout << "dummyTestDataSet_Ys: \n" << aNet->dummyTestDataSet_Ys[i].format(CleanFmt);
		std::cout << "\n\n";
	}*/

	aNet->classify();

	std::cin.get();

	return 0;
}




