#include "RNN.h"
#include <armadillo>


RNN_LSTM::RNN_LSTM(){

	
	numHiddenLayers = 

	std::vector<BaseLayer> inGateLayers, forgetGateLayers, cellStateLayers, outputGateLayers 
	std::vector<ElementWiseLayer>  outputLayers 	




}

RNN_LSTM::forward(){

	layerOutput.output->zeros();
        // to forward pass the Deep LSTM model, loop each time point, at each time, go through bottom layer to top layer
	for (int t = 0; t < T; t++)
		for (int l = 0; l < L; l++)

	if (l == 0){
		lowerLayer = dataLayer;
	} else {
		lowerLayer = layers[l-1];
	}
	
// concatenate to a large vector
	commonInput = [ lowerLayer.output ; layerOutput_prev[l].output]  ;
//1
	inGateLayers.input = commonInput 
 inGateLayers.activatUp();
//2
 InformationLayers[l].input = commonInput
	InformationLayers[l].activateUp();
//3
 inputElementGateLayers[l].inputOne = InformationLayers.output;
	inputElementGateLayers[l].inputTwo = inGateLayers.output;
	inputElementGateLayers[l].activateUp();
	
//4
 forgetGateLayers[l].input = commonInput;
	forgetGateLayers[l].activateUp();
//5
 forgetElementGate[l].inputOne = forgetGate.output;
	forgetElementGate[l].inputTwo = cellStateLayer.output;
	forgetElementGate[l].activateUp();
//6
 cellState[l].input = inputElementGate.output + forgetElementGate.output;
	cellState_prev[l].input = cellState[l].input; 
	cellState[l].activateUp();
//7
 outputGate[l].input = commonInput;
	outputGate[l].activateUp();
//8
 outputLayer[l].inputOne = outputGate.output;
	outputLayer[l].inputTwo = cellState.output;
	outputLayer[l].activateUp(); 

	layerOutput_prev[l] = layerOutput[l];
		}
	}

}

RNN_LSTM::backward(){
	layerOutput.output->zeros();

        // to backprop or backpass the Deep LSTM, start from the top layer of the last time point T, 
           // and then go through from top to bottom, and then go to previous time point, and loop 
            // from top to bottom layers again
	for (int t = T-1; t >= T; t++)
		for (int l = L-1; l >= 0; l++)

	if (l == 0){
		lowerLayer = dataLayer;
	} else {
		lowerLayer = layers[l-1];
	}
	
	delta = y - y_target;
        // this layer's output error comes from last time's (1)inGate, (2)g, (4)forgetGate, (7)outputGate, since output of each hidden layer will
          // be the input for the inGate, g, forgetGate, outputGate
        delta_prev = inGate_next.delta + Information_next.delta + forgetGate_next.delta + outputGate_next.delta;
	
	delta += delta_prev

	outputLayer[l].updatePara(delta);
	outputGate[l].updatePara(outputLayer.deltaoutOne);
//	cellSate[l].deltaOut +=cellState_next[l].deltaOut;
//	cellSate[l].deltaOut +=forgetElementGate_prev[l].deltaOut;
	cellState[l].updatePara(outputLayer.deltaoutTwo);
	inputElementGate.updatePara(cellState.deltaOut);
	
		}
	}





}



RNN::RNN(std::shared_ptr<arma::mat> trainingX0, std::shared_ptr<arma::mat> trainingY0,
       int inputDim0, int outputDim0, int numCells0, TrainingPara trainingPara0){
		   
	trainingX = trainingX0;
	trainingY = trainingY0;
	
	inputDim = inputDim0;
	outputDim = outputDim0;
	numCells = numCells;
	trainingPara = trainingPara0;
	
}

vod RNN::initializeWeight(){
	
	Y_iGate = std::make_shared<arma::mat>(numCells, trainingPara.miniBatchSize);
	Y_cCandit = std::make_shared<arma::mat>(numCells, trainingPara.miniBatchSize);
	Y_fGate = std::make_shared<arma::mat>(numCells,trainingPara.miniBatchSize);
	Y_oGate = std::make_shared<arma::mat>(numCells,trainingPara.miniBatchSize);
	S_c = std::make_shared<arma::mat>(numCells,trainingPara.miniBatchSize);
	S_cPrev = std::make_shared<arma::mat>(numCells,trainingPara.miniBatchSize);
	Y_c = std::make_shared<arma::mat>(numCells,trainingPara.miniBatchSize);
	
	W_iGate_x = std::make_shared<arma::mat>(iGateDim, inputDim);
	W_iGate_Sc = std::make_shared<arma::mat>(iGateDim, Dim);
	W_iGate_Yc = std::make_shared<arma::mat>(iGateDim, outputYcDim);
	W_Sc_x = std::make_shared<arma::mat>(numCells, inputDim);
	W_Sc_Yc = std::make_shared<arma::mat>(numCells, numCells);
	W_fGate_x = std::make_shared<arma::mat>(numCells, inputDim);
	W_fGate_Yc = std::make_shared<arma::mat>(numCells, numCells);
	W_fGate_Sc = std::make_shared<arma::mat>(numCells, numCells);
	W_oGate_x = std::make_shared<arma::mat>(numCells, inputDim);
	W_oGate_Yc = std::make_shared<arma::mat>(numCells, numCells);
	W_oGate_Sc = std::make_shared<arma::mat>(numCells, numCells);
	
	// initialize the derivatives to be 0
	ds_iGate_x = std::make_shared<arma::mat>(numCells, inpuDim, arma::fill::zeros);
	ds_iGate_Yc = std::make_shared<arma::mat>(numCells, numCells, arma::fill::zeros);
	ds_iGate_Sc = std::make_shared<arma::mat>(numCells, numCells, arma::fill::zeros);
	ds_fGate_x = std::make_shared<arma::mat>(numCells, inputDim, arma::fill::zeros);
	ds_fGate_Yc = std::make_shared<arma::mat>(numCells, numCells, arma::fill::zeros);
	ds_fGate_Sc = std::make_shared<arma::mat>(numCells, numCells, arma::fill::zeros);
	ds_Sc_x = std::make_shared<arma::mat>(numCells, inputDim,arma::fill::zeros);
	ds_Sc_Yc = std::make_shared<arma::mat>(numCells, numCells,arma::fill::zeros);
	ds_bias = std::make_shared<arma::vec>(numCells,arma::fill::zeros);
	
	ds_iGate_x_prev = std::make_shared<arma::mat>(numCells, inputDim, arma::fill::zeros);
	ds_iGate_Yc_prev = std::make_shared<arma::mat>(numCells, numCells, arma::fill::zeros);
	ds_iGate_Sc_prev = std::make_shared<arma::mat>(numCells, numCells, arma::fill::zeros);
	ds_fGate_x_prev = std::make_shared<arma::mat>(numCells, numCells, arma::fill::zeros);
	ds_fGate_Yc_prev = std::make_shared<arma::mat>(numCells, numCells, arma::fill::zeros);
	ds_fGate_Sc_prev = std::make_shared<arma::mat>(numCells, numCells, arma::fill::zeros);
	ds_Sc_x_prev = std::make_shared<arma::mat>(numCells, inputDim, arma::fill::zeros);
	ds_Sc_Yc_prev = std::make_shared<arma::mat>(numCells, numCells, arma::fill::zeros);
	ds_bias_prev = std::make_shared<arma::vec>(numCells, arma::fill::zeros);
	
}

void RNN::forwardPass(std::shared_ptr<arma::mat> X) {
	// X is the input from training or tesing samples
	/*
	  LSTM forwardpass
	*/
	
	//kl forwardpass: 
		
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
    (*Y_oGate) = (*W_oGate_x) * (*X).st() +（*W_oGate_Yc）* (*Y_cPrev).st() ＋ (*W_oGate_Sc) * (*S_cPrev).st(); // net_outj(t) in kl deduction
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
	
    outLayer.activateUp(outputLayer.W * (*Y_c)); // Yk = fk(net_k(t)) = fk(W_k_m * ms(t))
    /*
    	*Y = (*H).st() * (*W_oh).st() ;
    	for (int i = 0; i < Y->n_rows; i++)
    		Y->row(i) += (*B_o).st();
    */
	
	// derivatives for further use in backwardpass
	arma::mat delta_output_temp = (*outLayer.outputY) - (*trainingY);
    arma::vec delta_output = arma::sum(delta_output_temp,1);
// calcuate the derivatives respect to internal states
    arma::mat H_cDeriv = (1 - (*H_c) % (*H_c)); // h'(S_cjv(t)) in kl deduction, tanh derivative

	// internal state error
    arma::mat e_Sc = (*Y_oGate) * (H_cDeriv) * (*outLayer.W) * delta_output; // e_scjv(t) in kl deduction P2, outputLayer.W is the W_km 

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


    *ds_bias = *ds_bias_prev * (*Y_fGate);
}

void RNN::updatePara() {

   
	/*
	  LSTM backpropagation weight update for a sample sequence 
	*/
	
    //*B_oGate -= *B_oGate_accu;
	
	// output gate weight update
    *W_oGate_Yc -= *W_oGate_Yc_accu;
    *W_oGate_Sc -= *W_oGate_Sc_accu;

	// input gate weight update
    *W_iGate_x -= *W_iGate_x_accu;
	*W_iGate_Yc -= *W_iGate_Yc_accu;
	*W_iGate_Sc -= *W_iGate_Sc_accu;	

	// forget gate weight update
    *W_fGate_x -= *W_fGate_x_accu;
	*W_fGate_Yc -= *W_fGate_Yc_accu;
	*W_fGate_Sc -= *W_fGate_Sc_accu;	
	
	// cell weight update
    *W_Sc_x -= *W_Sc_x_accu;
    *W_Sc_Yc -= *W_Sc_Yc_accu;


}

void RNN::backwardPass(std::shared_ptr<arma::mat> delta_output, double learningRate) {

   
	/*
	  LSTM backpropagation with truncation, no error backtransfered from t to t-1
	*/
	//backprop:

	double alpha = learningRate;
// update the output weight and bias, same as MLP
    *(outLayer.W) -= learningRate * delta_output * (*outLayer.inputX);
    *(outLayer.B) -= learningRate * delta_output;


// now update the output gate Weight and bias
// W_oGate_i
	std::shared_ptr<arma::mat> deriv(new arma::mat(numCells,));
    *deriv =  (*Y_oGate) % (1 - *Y_oGate);
	std::shared_ptr<arma::mat> delta_oGate(new arma::mat(,));
    *delta_oGate = deriv * (*H_c) * (*outLayer.W) * delta_output;
	std::shared_ptr<arma::mat> grad(new arma::mat(,));
// oGate's input is X
    *grad = (*delta_oGate) * (*X)
    *W_oGate_i -= alpha * (*grad);
    *B_oGate -= alpha * (*delta_oGate);
// W_oGate_Yc
    *grad = (*delta_oGate) * (*Y_cPrev);
    *W_oGate_Yc_accu -= alpha * (*grad);
	// W_oGate_Sc
    *grad = (*delta_oGate) * (*S_cPrev);
    *W_oGate_Sc_accu -= alpha * (*grad);

// update the in gate weight and bias
//	W_iGate_x
    *grad = e_Sc * (*ds_iGate_x);
    *W_iGate_x_accu -= alpha * (*grad);
// W_iGate_Yc
    *grad = e_Sc * (*ds_iGate_Yc);
    *W_iGate_Yc_accu -= alpha * (*grad);
// W_iGate_Sc
    *grad = e_Sc * (*ds_iGate_Sc);
    *W_iGate_Sc_accu -= alpha * (*grad);	

// update the forget gate weight and bias
// W_fGate_x
    *grad = e_Sc * (*ds_fGate_x);
    *W_fGate_x_accu -= alpha * (*grad);
// W_fGate_Yc
    *grad = e_Sc * (*ds_fGate_Yc);
    *W_fGate_Yc_accu -= alpha * (*grad);
// W_fGate_Sc
    *grad = e_Sc * (*ds_fGate_Sc);
    *W_fGate_Sc_accu -= alpha * (*grad);	
	
// update the cell weight and bias
// W_Sc_i
    *grad = e_Sc * (*ds_Sc_x);
    *W_Sc_x_accu -= alpha * (*grad);
// W_Sc_Yc
    *grad = e_Sc * (*ds_Sc_Yc);
    *W_Sc_Yc_accu -= alpha * (*grad);

//	update the cell bias
    //*B_c -= e_Sc * (*ds_bias);

}

void RNN::train(){
	/*
	  LSTM training 
	*/
    std::shared_ptr<arma::mat> subInputX = std::make_shared<arma::mat>();
    std::shared_ptr<arma::mat> subInputY = std::make_shared<arma::mat>();
    std::shared_ptr<arma::mat> delta = std::make_shared<arma::mat>();
	
    int ntimes; 
    double error, errorTotal;
    int size = trainingPara.miniBatchSize; // a minibatch could be a sentence of words
    for (int epoch = 0; epoch < trainingPara.NEpoch; epoch++) { // each epoch will run the whole training data once
        std::cout << epoch << std::endl;
        ntimes  = numInstance / trainingPara.miniBatchSize;
        errorTotal = 0.0;
		// we assume each sample spanned T length (T time points)for X and Y, T length of time series
		// each X will be forwardpass to calculate each gate and derivatives, 
		 // and Y is used to parallel backpropagate between two adjacent time points for the whole length of time series
		 // as a result, forwardpass and backwardpass will be implemented alternatively to calculate weight update for each time point
		// for loop for the whole samples, with each i for a minibatch of samples
        for (int i = 0; i < ntimes; i++) {
            (*subInputX) = trainingX->rows(i*size,(i+1)*size-1);
            (*subInputY) = trainingY->rows(i*size,(i+1)*size-1);            
            forwardPass(subInputX);
 //           output->print(); 
            (*delta) = ((*output) - (*subInputY)).st();
 //           subInputY->print();
            backwardPass(delta);            
            error = arma::sum(arma::sum((*delta).st() * (*delta)));
            errorTotal += error;                        
        }
		updatePara();
        std::cout << errorTotal << std::endl;
    }
	
}
