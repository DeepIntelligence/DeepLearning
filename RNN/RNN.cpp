#include "RNN.h"


RNN::forwardPass(std::shared_ptr<arma::mat> inputX) {

    *Y_iGate = (*W_iGate_i) * (*X).st() + (*W_iGate_c) * (*Y_cPrev).st();
    for (int i = 0; i < Y_iGate->n_rows; i++)
        Y_iGate->row(i) += (*B_iGate).st();
    Y_iGate->transform([](double val) {
        return 1.0/(1+exp(-val));
    });

    *Y_cCandit = (*W_c_i) * (*X) + (*W_c_c) * (*Y_cPrev);
    for (int i = 0; i < Y_cCandit->n_rows; i++)
        Y_cCandit->row(i) += (*B_c).st();
    Y_cCandit->transform([](double val) {
        return tanh(val);
    });

    *Y_fGate = (*W_fGate_i) * (*X) + (*W_fGate_c) * (*Y_cPrev);
    for (int i = 0; i < Y_fGate->n_rows; i++)
        Y_fGate->row(i) += (*B_fGate).st();
    Y_fGate->transform([](double val) {
        return 1.0/(1+exp(-val));
    });


    (*Y_oGate) = (*W_oGate_i) * (*X) + (*W_oGate_c) * (*Y_cPrev);
    for (int i = 0; i < Y_oGate->n_rows; i++)
        Y_oGate->row(i) += (*B_oGate).st();
    Y_oGate->transform([](double val) {
        return 1.0/(1+exp(-val));
    });

    *S_c = (*Y_iGate) % (*Y_cCandit) + (*Y_fGate) % (*S_cPrev);
    *S_cPrev = *S_c;
    *H_c = *S_c;
    H_c->transform([](double val) {
        return tanh(val);
    });
    *Y_cPrev = *Y_c;
    *Y_c = (*Y_oGate) % (*H_c);

    outLayer.activateUp(Y_c);
    /*
    	*Y = (*H).st() * (*W_oh).st() ;
    	for (int i = 0; i < Y->n_rows; i++)
    		Y->row(i) += (*B_o).st();
    */
}

RNN::backwardPass() {

    arma::mat delta_output_temp = (*outLayer.outputY) - (*trainingY);
    arma::vec delta_output = arma::sum(delta_output_temp,1);
// calcuate the derivatives respect to internal states
    arma::mat H_cDeriv = (1 - H_c % H_c);

    arma::mat e_Sc = Y_oGate * (H_cDeriv) * (*outLayer.W) * delta_output;

    ds_iGate_i = ds_iGate_i_prev * Y_fGate + Y_cCandit * (Y_iGate)* (1-Y_iGate) * X;
    ds_iGate_c = ds_iGate_c_prev * Y_fGate + Y_cCandit * (Y_iGate)* (1-Y_iGate) * Y_cPrev;

    ds_fGate_i = ds_fGate_i_prev * Y_fGate + S_cPrev * (Y_fGate)* (1-Y_fGate) * X;
    ds_fGate_c = ds_fGate_c_prev * Y_fGate + S_cPrev * (Y_fGate)* (1-Y_fGate) * Y_cPrev;

    ds_c_i = ds_c_i_prev * Y_fGate + Y_iGate * (Y_cPrev)* (1-Y_cPrev) * X;
    ds_c_c = ds_c_c_prev * Y_fGate + Y_iGate * (Y_cPrev)* (1-Y_cPrev) * Y_cPrev;


    ds_bias = ds_bias_prev * Y_fGate;


// update the output weight and bias, same as MLP
    *(outLayer.W) -= alpha * delta_output * (*outLayer.inputX);
    *(outLayer.B) -= alpha * delta_output;


// now update the output gate Weight and bias
// W_oGate_i
    deriv =  Y_oGate % (1 - Y_oGate)
             delta_oGate = deriv * (*H_c) * outLayer.W * delta_output;
// oGate's input is X
    grad = delta_oGate * (*X)
           *W_oGate_i -= alpha * grad;
    *B_oGate -= alpha * delta_oGate;
// W_oGate_c
    grad = delta_oGate * (*Y_cPrev);
    *W_oGate_c -= alpha *grad;

// update the in gate weight and bias
//	W_iGate_i
    grad = e_Sc * dS_iGate_i;
    W_iGate_i -= alpha * grad;
// W_iGate_c
    grad = e_Sc * ds_iGate_c;
    W_iGate_c -= alpha * grad;

// update the forget gate weight and bias
// W_fGate_i
    grad = e_Sc * ds_fGate_i;
    W_fGate_c -= alpha * grad;
// W_fGate_c
    grad = e_Sc * ds_fGate_c;
    W_fGate_c -= alpha * grad;
// update the cell weight and bias
// W_c_i
    grad = e_Sc * ds_c_i;
    W_c_i -= alpha * grad;
// W_c_c
    grad = e_Sc * ds_c_c;
    W_c_c = alpha *grad;

//	update the cell bias
    B_c -= e_Sc * ds_bias;


}