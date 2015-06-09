#include <memory>




struct InputLayer {};
struct LSTM {};
struct OutputLayer {};


class RNN {





public:
	
  // methods
    void forwardPass();
    void backwardPass();
  
  // attributes
	//internal states  
    std::shared_ptr<arma::mat> Y_iGate, Y_cCandit, Y_fGate, Y_oGate, S_c, S_cPrev, H_c, Y_c;
	// weights
    std::shared_ptr<arma::mat> W_iGate_x, W_iGate_Sc, W_iGate_Yc;
	std::shared_ptr<arma::mat> W_Sc_x, W_Sc_Yc;
	std::shared_ptr<arma::mat> W_fGate_x, W_fGate_Yc, W_fGate_Sc;
	std::shared_ptr<arma::mat> W_oGate_x, W_oGate_Yc, W_oGate_Sc;
	// derivatives
	std::shared_ptr<arma::mat> ds_iGate_x, ds_iGate_Yc, ds_iGate_Sc, ds_iGate_x_prev, ds_iGate_Yc_prev, ds_iGate_Sc_prev;
	std::shared_ptr<arma::mat> ds_fGate_x, ds_fGate_Yc, ds_fGate_Sc, ds_fGate_x_prev, ds_fGate_Yc_prev, ds_fGate_Sc_prev;
	std::shared_ptr<arma::mat> ds_Sc_x, ds_Sc_Yc, ds_Sc_x_prev, ds_Sc_Yc_prev;
	std::shared_ptr<arma::mat> ds_bias, ds_bias_prev; 
	
    std::shared_ptr<arma::mat> U_ci, U_cf, U_cc, U_io, V_o;
    std::shared_ptr<arma::mat> trainingX, trainingY;
    std::shared_ptr<arma::mat> inputX;
    std::shared_ptr<arma::mat> H, C, H_prev;
    std::shared_ptr<arma::vec> B_igate, B_ogate, B_fgate;
	int inputDim, outputDim, T, numCells; 
	BaseLayer outLayer;
		
		

};

