#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    /* TODO: Calculate the RMSE here.*/

    // Initialize RMSE Vector
    VectorXd rmse = VectorXd::Zero(4);

    // Check Input Vectors
    if(estimations.size() == 0 || estimations.size() != ground_truth.size()){
        cout << "Error in inputs" << endl;
        return rmse;
    }
    // Accumulate Squared Residuals
    VectorXd residuals(4);

    for(int i=0; i < estimations.size(); i++){
        residuals = estimations[i] - ground_truth[i];
        residuals = residuals.array() * residuals.array();
        rmse = rmse + residuals;
    }

    // Calculate the mean
    rmse = rmse/estimations.size();
    // Calculate the squared root
    rmse = rmse.array().sqrt();
    // Return Output
    return rmse;
}
