#include "RNN.h"


void RNN::forwardPass(std::shared_ptr<arma::mat> inputX) {
	/*
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
	*/
	
	/*
	  LSTM forwardpass
	*/
	
	//kl forwardpass: 
	// initialize for time series recursion ??
		
	// input gate
	// Y_cPrev = ms(t) in kl deduction
    *Y_iGate = (*W_iGate_x) * (*X).st() + (*W_iGate_Sc) * (*S_cPrev).st() + (*W_iGate_Yc)*(*Y_cPrev).st(); // net_inj(t) in kl deduction
    for (int i = 0; i < Y_iGate->n_rows; i++)
        Y_iGate->row(i) += (*B_iGate).st(); 
    Y_iGate->transform([](double val) {   // y_inj(t) in kl deduction
        return 1.0/(1+exp(-val));  // f_inj takes sigmoid function
    });

    // cell candidate state
    *Y_cCandit = (*W_Sc_x) * (*X).st() + (*W_Sc_Yc) * (*Y_cPrev).st(); // net_cjv(t) in kl deduction
    for (int i = 0; i < Y_cCandit->n_rows; i++)
        Y_cCandit->row(i) += (*B_c).st();
    Y_cCandit->transform([](double val) {   // g(net_cjv(t)) in kl deduction
        return tanh(val);    // g taks tanh function
    });

	// forget gate
    *Y_fGate = (*W_fGate_x) * (*X).st() + (*W_fGate_Yc)*(*Y_cPrev).st() + (*W_fGate_Sc) * (*S_cPrev).st(); // net_phi(t) in kl deduction
    for (int i = 0; i < Y_fGate->n_rows; i++)
        Y_fGate->row(i) += (*B_fGate).st();
    Y_fGate->transform([](double val) {   // y_phi(t) in kl deduction
        return 1.0/(1+exp(-val)); // f_phi_j in kl deduction
    });

	// output gate
    (*Y_oGate) = (*W_oGate_x) * (*X).st() +（*W_oGate_Yc）* (*Y_cPrev).st() ＋ (*W_oGate_Yc) * (*Y_cPrev).st(); // net_outj(t) in kl deduction
    for (int i = 0; i < Y_oGate->n_rows; i++)
        Y_oGate->row(i) += (*B_oGate).st();
    Y_oGate->transform([](double val) { // y_outj(t) in kl deduction
        return 1.0/(1+exp(-val)); // f_outj in kl deduction
    });

	// cell state based on input gate and forget gate
    *S_c = (*Y_iGate) % (*Y_cCandit) + (*Y_fGate) % (*S_cPrev); // S_cjv(t) in kl deduction
    *S_cPrev = *S_c;
    *H_c = *S_c;  // h(S_cjv(t)) in kl figure
    H_c->transform([](double val) {
        return tanh(val); // h takes tanh function
    });
    
    *Y_c = (*Y_oGate) % (*H_c); // Y_c = ms(t) in kl deduction
    *Y_cPrev = *Y_c;  
	
    outLayer.activateUp(outputLayer.W * Y_c); // Yk = fk(net_k(t)) = fk(W_k_m * ms(t))
    /*
    	*Y = (*H).st() * (*W_oh).st() ;
    	for (int i = 0; i < Y->n_rows; i++)
    		Y->row(i) += (*B_o).st();
    */
}


void RNN::backwardPass() {

    /*
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
*/
	
	/*
	  LSTM backpropagation
	*/
	//kl backprop:
	arma::mat delta_output_temp = (*outLayer.outputY) - (*trainingY);
    arma::vec delta_output = arma::sum(delta_output_temp,1);
// calcuate the derivatives respect to internal states
    arma::mat H_cDeriv = (1 - H_c % H_c); // h'(S_cjv(t)) in kl deduction, tanh derivative

	// internal state error
    arma::mat e_Sc = Y_oGate * (H_cDeriv) * (*outLayer.W) * delta_output; // e_scjv(t) in kl deduction P2, outputLayer.W is the W_km 

	// input gate weigth derivative, including d_W_i_x, d_W_i_Yc(d_W_i_ms), d_W_i_Sc
    *ds_iGate_x = (*ds_iGate_x_prev) * (*Y_fGate) + (*Y_cCandit) * (*Y_iGate)* (1-*Y_iGate) * (*X); // d_W_i_x
    *ds_iGate_Yc = (*ds_iGate_Yc_prev) * (*Y_fGate) + (*Y_cCandit) * (*Y_iGate)* (1-*Y_iGate) * (*S_cPrev); // d_W_i_Sc
	*ds_iGate_Sc = (*ds_iGate_Sc_prev) * (*Y_fGate) + (*Y_cCandit) * (*Y_iGate)* (1-*Y_iGate) * (*Y_cPrev); // d_W_i_Yc
	
	// forget gate weigth derivative, including d_W_f_x, d_W_f_Yc (d_W_f_ms), d_W_f_Sc
    *ds_fGate_x = (*ds_fGate_x_prev) * (*Y_fGate) + (*S_cPrev) * (*Y_fGate)* (1-*Y_fGate) * (*X); // d_W_f_x 
    *ds_fGate_Yc = (*ds_fGate_Yc_prev) * (*Y_fGate) + (*S_cPrev) * (*Y_fGate)* (1-*Y_fGate) * (*Y_cPrev); // d_W_f_Yc
	*ds_fGate_Sc = (*ds_fGate_Sc_prev) * (*Y_fGate) + (*S_cPrev) * (*Y_fGate)* (1-*Y_fGate) * (*S_cPrev); // d_W_f_Sc

	// cell input weigth derivative, including d_W_Sc_x, d_W_Sc_Yc
    *ds_Sc_x = (*ds_Sc_x_prev) * (*Y_fGate) + (*Y_iGate) * (*Y_cPrev)* (1-*Y_cPrev) * (*X); // d_W_Sc_x
    *ds_Sc_Yc = (*ds_Sc_Yc_prev) * (*Y_fGate) + (*Y_iGate) * (*Y_cPrev)* (1-*Y_cPrev) * (*Y_cPrev); // d_W_Sc_Yc


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
// W_oGate_Yc
    grad = delta_oGate * (*Y_cPrev);
    *W_oGate_Yc -= alpha *grad;
	// W_oGate_Sc
    grad = delta_oGate * (*S_cPrev);
    *W_oGate_Sc -= alpha *grad;

// update the in gate weight and bias
//	W_iGate_x
    grad = e_Sc * ds_iGate_x;
    W_iGate_i -= alpha * grad;
// W_iGate_Yc
    grad = e_Sc * ds_iGate_Yc;
    W_iGate_Yc -= alpha * grad;
// W_iGate_Sc
    grad = e_Sc * ds_iGate_Sc;
    W_iGate_Sc -= alpha * grad;	

// update the forget gate weight and bias
// W_fGate_x
    grad = e_Sc * ds_fGate_x;
    W_fGate_x -= alpha * grad;
// W_fGate_Yc
    grad = e_Sc * ds_fGate_Yc;
    W_fGate_Yc -= alpha * grad;
// W_fGate_Sc
    grad = e_Sc * ds_fGate_Sc;
    W_fGate_Sc -= alpha * grad;	
	
// update the cell weight and bias
// W_Sc_i
    grad = e_Sc * ds_Sc_i;
    W_c_x -= alpha * grad;
// W_Sc_Yc
    grad = e_Sc * ds_Sc_Yc;
    W_Sc_Yc = alpha *grad;

//	update the cell bias
    B_c -= e_Sc * ds_bias;

}

void RNN::train(){
	/*
	  LSTM training 
	*/
    std::shared_ptr<arma::cube> subInput = std::make_shared<arma::cube>();
    std::shared_ptr<arma::mat> subInputY = std::make_shared<arma::mat>();
    std::shared_ptr<arma::mat> delta = std::make_shared<arma::mat>();
    int ntimes;
    double error, errorTotal;
    int size = trainingPara.miniBatchSize;
    for (int epoch = 0; epoch < trainingPara.NEpoch; epoch++) {
        std::cout << epoch << std::endl;
        ntimes  = numInstance / trainingPara.miniBatchSize;
        errorTotal = 0.0;
        for (int i = 0; i < ntimes; i++) {
            (*subInput) = trainingX->slices(i*size*nChanel,(i+1)*size*nChanel-1);
            (*subInputY) = trainingY->rows(i*size,(i+1)*size-1);            
            feedForward(subInput);
 //           output->print();
            (*delta) = ((*output) - (*subInputY)).st();
 //           subInputY->print();
            backProp(delta);            
            error = arma::sum(arma::sum((*delta).st() * (*delta)));
            errorTotal += error;                        
        }
        std::cout << errorTotal << std::endl;
    }
	
}
