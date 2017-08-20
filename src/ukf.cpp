#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  1. Initializes Unscented Kalman Filter
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// UKF Constructor
UKF::UKF() {
    is_initialized_ = false;
    time_us_ = 0;

    // If this is false, laser measurements will be ignored (except during init)
    use_laser_ = false;

    // If this is false, radar measurements will be ignored (except during init)
    use_radar_ = false;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = .5;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 3;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    // Set state dimension
    n_x_ = 5;

    // Set augmented dimension
    n_aug_ = 7;

    // Initial state vector
    x_ = VectorXd(5);

    // Initial covariance matrix
    P_ = MatrixXd(5 , 5);
    P_.setIdentity();

    // Define spreading parameter
    lambda_ = 3 - n_aug_;

    Xsig_pred_ = MatrixXd(5 , 2 * n_aug_ + 1);

    // Initialize weight
    weights_ = VectorXd(2*n_aug_+1);
    // Set first weight
    double weight_0 = lambda_/(lambda_+n_aug_);
    weights_(0) = weight_0;
    // Set remaining weights
    for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
        double weight = 0.5/(n_aug_+lambda_);
        weights_(i) = weight;
    }
}

UKF::~UKF() {}

//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  2. Unscented Kalman Filter Process Measurement Step
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    if (!is_initialized_) {
        cout <<"initializing..."<<endl;
        // Determine which sensor to use
        // RADAR
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            double rho      = meas_package.raw_measurements_(0);
            double theta    = meas_package.raw_measurements_(1);
            double rho_dot  = meas_package.raw_measurements_(2);
            x_ << rho * cos(theta) , rho * sin(theta), rho_dot, 0, 0;
        }
        // LIDAR
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            double x = meas_package.raw_measurements_(0);
            double y = meas_package.raw_measurements_(1);
            x_ << x, y, 0, 0,0;
        }
        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        return;
    }
    double dt = (meas_package.timestamp_ - time_us_) *1e-6;	// converted to seconds
    time_us_ = meas_package.timestamp_;

    // RADAR
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
        use_radar_ = true;
    // LIDAR
    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
        use_laser_ = true;

    Prediction(dt);

    // RADAR
    if (use_radar_ ){
        UpdateRadar(meas_package);
        use_radar_ = false;
    }
    // LIDAR
    if (use_laser_ ){
        UpdateLidar(meas_package);
        use_laser_ = false;
    }
}

//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  3. Unscented Kalman Filter Prediction Step
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void UKF::Prediction(double delta_t) {
    // Create Predicted Sigma Point Matrix (5x15)
    MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);
    MatrixXd A = P_.llt().matrixL();
    Xsig.col(0)  = x_;
    for (int i = 0; i < n_x_; i++){
        Xsig.col(i+1)      = x_ + sqrt(lambda_+n_x_) * A.col(i);
        Xsig.col(i+1+n_x_) = x_ - sqrt(lambda_+n_x_) * A.col(i);
    }

    // Create Augmented Mean Vector (7-D)
    VectorXd x_aug = VectorXd(n_aug_);
    // Create Augmented State Covariance Matrix (7x7)
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    // Create Sigma Point Matrix (7x15)
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    // Create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    // Create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = P_;
    P_aug(5,5) = std_a_*std_a_;
    P_aug(6,6) = std_yawdd_*std_yawdd_;

    // Create the square root matrix
    MatrixXd L = P_aug.llt().matrixL();
    // Create the mean sigma points
    Xsig_aug.col(0)  = x_aug;
    // Create sqrt_function for easier readability
    double sqrt_function = sqrt(lambda_+n_aug_);
    for (int i = 0; i< n_aug_; i++){
        Xsig_aug.col(i+1)         = x_aug + sqrt_function * L.col(i);
        Xsig_aug.col(i+1+n_aug_)  = x_aug - sqrt_function * L.col(i);
    }

    // Asign values to each value in the matrix
    for (int i = 0; i< 2*n_aug_+1; i++){
        // Extract values for better readability
        double p_x = Xsig_aug(0,i);       // P_x value
        double p_y = Xsig_aug(1,i);       // P_y value
        double v = Xsig_aug(2,i);         // v value
        double yaw = Xsig_aug(3,i);       // yaw rate value
        double yawd = Xsig_aug(4,i);      // yaw rate dot value
        double nu_a = Xsig_aug(5,i);      // longitudinal acceleration noise
        double nu_yawdd = Xsig_aug(6,i);  // yaw acceleration noise

        // Predicted state values for x and y
        double px_p, py_p;

        // Avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        } else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;

        // Add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;
        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;

        // Write predicted sigma point into proper columns
        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = yaw_p;
        Xsig_pred_(4,i) = yawd_p;
    }

    // Update state (x) and covariance (P)

    // Predicted State Mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  // Iterate over sigma points
        x_ = x_+ weights_(i) * Xsig_pred_.col(i);
    }

    // Predicted State Covariance Matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        // State difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        // Angle normalization
        x_diff(3)= Normalize_Function(x_diff(3));
        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }
}

//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  4. Unscented Kalman Filter Update Lidar Step
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void UKF::UpdateLidar(MeasurementPackage meas_package) {
    // Assign incoming LIDAR measurement to z variable
    VectorXd z = meas_package.raw_measurements_;

    // Create R_ and H_ Matrices
    MatrixXd H_ = MatrixXd(2, 5); // constant
    H_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0;
    MatrixXd R_ = MatrixXd(2, 2);
    R_ << std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;

    // Measurement Covariance Matrix for Lidar
    VectorXd y = z - H_ * x_;
    MatrixXd S = H_ * P_ * H_.transpose() + R_;
    MatrixXd K = P_ * H_.transpose() * S.inverse();

    // New Estimates
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  5. Unscented Kalman Filter Update Radar Step
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void UKF::UpdateRadar(MeasurementPackage meas_package) {
    // Set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;

    // Create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    // Transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        // Extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);
        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        // Measurement model
        Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);   //r
        Zsig(1,i) = atan2(p_y,p_x);   //phi
        Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }

    // Mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    // Measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        // Residual vector
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // Angle normalization
        z_diff(1) = Normalize_Function(z_diff(1));
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
    // Add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<    std_radr_ * std_radr_, 0, 0,
            0, std_radphi_*std_radphi_, 0,
            0, 0,std_radrd_*std_radrd_;
    S = S + R;

    // Assign incoming RADAR measurement to z variable
    VectorXd z = meas_package.raw_measurements_;

    // Create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    // Calculate cross correlation matrix
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        // Residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // Angle normalization
        z_diff(1) = Normalize_Function(z_diff(1));

        // State difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //Angle normalization
        x_diff(3) = Normalize_Function(x_diff(3));

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }
    // Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    // Residual
    VectorXd z_diff = z - z_pred;

    // Angle normalization
    z_diff(1) = Normalize_Function(z_diff(1));

    // Update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();
}

double UKF::Normalize_Function(double x) {
  while(x < -M_PI) x += 2 * M_PI;
  while(x >  M_PI) x -= 2 * M_PI;
  return x;
}
