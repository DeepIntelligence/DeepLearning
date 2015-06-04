
#include "CNN.h"


CNN::CNN(std::shared_ptr<arma::cube> trainingX0, std::shared_ptr<arma::mat> trainingY0, int nChanel0){
    trainingX = trainingX0;
    trainingY = trainingY0;
    nChanel = nChanel0;
    numInstance = trainingX->n_slices / nChanel;
    inputDim_x = trainingX->n_rows;
    inputDim_y = trainingX->n_cols;

    convoLayers.push_back(ConvolveLayer(10,9,9,1));
    convoLayers[0].setInputDim(inputDim_x, inputDim_y, nChanel);
//    convoLayers.push_back(ConvolveLayer(5,2,2,1));
    poolLayers.push_back(PoolLayer(2,2,PoolLayer::mean));
    poolLayers[0].setInputDim(convoLayers[0].outputDim_x,
                              convoLayers[0].outputDim_y,
                              convoLayers[0].outputDim_z);
    
    int totalSize = poolLayers[0].outputDim_x * poolLayers[0].outputDim_y * poolLayers[0].outputDim_z;
    
//    poolLayers.push_back(PoolLayer(2,2,PoolLayer::mean));
                            
    FCLayers.push_back(BaseLayer(totalSize, 10,BaseLayer::softmax));
//    FCLayers.push_back(BaseLayer(100,10,BaseLayer::softmax));

}

void CNN::setTrainingData(std::shared_ptr<arma::cube> trainingX0, std::shared_ptr<arma::mat> trainingY0, int nChanel0){
    trainingX = trainingX0;
    trainingY = trainingY0;
    nChanel = nChanel0;
    numInstance = trainingX->n_slices;
    inputDim_x = trainingX->n_rows;
    inputDim_y = trainingX->n_cols;
    
}

void CNN::train() {
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
            (*delta) = ((*output) - (*subInputY)).st();
            backProp(delta);            
            error = arma::sum(arma::sum((*delta).st() * (*delta)));
            errorTotal += error;                        
        }
        std::cout << error << std::endl;
    }
}

void CNN::feedForward(std::shared_ptr<arma::cube> subInput0) {

    std::shared_ptr<arma::cube> subInput = subInput0;
    for (int i = 0; i < numCLayers; i++) {
        if ( i == 0 ) {
            convoLayers[i].activateUp(subInput);
            subInput = convoLayers[i].output;
            poolLayers[i].activateUp(subInput);           
        }
        subInput = poolLayers[i-1].output;
        convoLayers[i].activateUp(subInput);
        subInput = convoLayers[i].output;
        poolLayers[i].activateUp(subInput);
    }
    
    subInput = poolLayers[numCLayers-1].output;
    int totalSize = poolLayers[numCLayers].outputDim_x * poolLayers[numCLayers].outputDim_y * poolLayers[numCLayers].outputDim_z; 
    subInput->reshape(totalSize,1,1);
    std::shared_ptr<arma::mat> subInput_mat = std::make_shared<arma::mat>(totalSize,1);
    (*subInput_mat) = subInput->slice(0);
    
    for (int i = 0; i < numFCLayers; i++) {
        FCLayers[0].activateUp(subInput_mat);
        subInput_mat = FCLayers[0].outputY;
    }
//    arma::mat delta_target = *(baseLayers[i-1].outputY) - *(trainingY);
    output = FCLayers[numFCLayers-1].outputY;
}

void CNN::backProp(std::shared_ptr<arma::mat> delta_target){
    std::shared_ptr<arma::mat> delta_in = delta_target;
    double learningRate = trainingPara.alpha / trainingPara.miniBatchSize;
    for (int i = numFCLayers-1; i >= 0 ; i--){
        FCLayers[i].updatePara(delta_in, learningRate);
        delta_in = FCLayers[i].delta_out;        
    }
//  remember to transform to 3d      
    std::shared_ptr<arma::cube> delta_in3D;
    for (int i = numCLayers - 1; i >= 0 ; i--){
        poolLayers[i].upSampling(delta_in3D);
        delta_in3D = poolLayers[i].delta_out;
        convoLayers[i].updatePara(delta_in3D, learningRate);
        delta_in3D = convoLayers[i].delta_out;
    }
}