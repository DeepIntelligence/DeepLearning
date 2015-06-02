#include <memory>




struct InputLayer {};
struct LSTM {};
struct OutputLayer {};


class RNN {







    void forwardPass();
    void backwardPass();


    std::shared_ptr<arma::mat> W_if, W_ic, W_ii, W_io;
    std::shared_ptr<arma::mat> U_ci, U_cf, U_cc, U_io, V_o;
    std::shared_ptr<arma::mat> trainingX, trainingY;
    std::shared_ptr<arma::mat> inputX;
    std::shared_ptr<arma::mat> H, C, H_prev;
    std::shared_ptr<arma::vec> B_igate, B_ogate, B_fgate;

};

