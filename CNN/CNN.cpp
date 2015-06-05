
#include "CNN.h"


CNN::CNN(std::shared_ptr<arma::cube> trainingX0, std::shared_ptr<arma::mat> trainingY0, int nChanel0, TrainingPara trainingPara0){
    trainingX = trainingX0;
    trainingY = trainingY0;
    nChanel = nChanel0;
    numInstance = trainingX->n_slices / nChanel;
    inputDim_x = trainingX->n_rows;
    inputDim_y = trainingX->n_cols;
    
    trainingPara = trainingPara0;

    convoLayers.push_back(ConvolveLayer(10,9,9,1));
    convoLayers[0].setInputDim(inputDim_x, inputDim_y, nChanel);
//    convoLayers.push_back(ConvolveLayer(5,2,2,1));
    poolLayers.push_back(PoolLayer(2,2,PoolLayer::mean));
    poolLayers[0].setInputDim(convoLayers[0].outputDim_x,
                              convoLayers[0].outputDim_y,
                              convoLayers[0].outputDim_z);
    
                            
    FCLayers.push_back(BaseLayer(poolLayers[0].outputSize, 10, BaseLayer::softmax));
//    FCLayers.push_back(BaseLayer(100,10,BaseLayer::softmax));
    numCLayers = 1;
    numFCLayers = 1;
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

void CNN::feedForward(std::shared_ptr<arma::cube> subInput0) {

    std::shared_ptr<arma::cube> subInput = subInput0;
    for (int i = 0; i < numCLayers; i++) {
        convoLayers[i].activateUp(subInput);
        subInput = convoLayers[i].output;
 //       subInput->save("convo_output.dat",arma::raw_ascii);
        poolLayers[i].activateUp(subInput);
        subInput = poolLayers[i].output;
 //       subInput->save("pool_output.dat",arma::raw_ascii);
    }
    int totalSize = poolLayers[numCLayers-1].outputSize; 
    subInput->reshape(1,totalSize,1);
    std::shared_ptr<arma::mat> subInput_mat = std::make_shared<arma::mat>(1,totalSize);
    (*subInput_mat) = subInput->slice(0);
//    subInput_mat->save("subinputmat.dat",arma::raw_ascii);
    for (int i = 0; i < numFCLayers; i++) {
        FCLayers[0].activateUp(subInput_mat);
        subInput_mat = FCLayers[0].outputY;
    }
//    arma::mat delta_target = *(baseLayers[i-1].outputY) - *(trainingY);
    output = FCLayers[numFCLayers-1].outputY;
//    output->save("networkoutput.dat", arma::raw_ascii);
}

void CNN::backProp(std::shared_ptr<arma::mat> delta_target){
    std::shared_ptr<arma::mat> delta_in = delta_target;
    delta_target->save("delta_target.dat",arma::raw_ascii);
    double learningRate = trainingPara.alpha / trainingPara.miniBatchSize;
    for (int i = numFCLayers-1; i >= 0 ; i--){
        FCLayers[i].updatePara(delta_in, learningRate);
        delta_in = FCLayers[i].delta_out;        
    }
//  remember to transform to 3d    ,
    
    std::shared_ptr<arma::cube> delta_in3D = std::make_shared<arma::cube>(poolLayers[numCLayers-1].outputDim_x,
            poolLayers[numCLayers-1].outputDim_y, poolLayers[numCLayers-1].outputDim_z);
    int count = 0;
    for (int i = 0; i <  poolLayers[numCLayers-1].outputDim_z; i++){
        for (int j = 0; j <  poolLayers[numCLayers-1].outputDim_y; j++){
            for (int k = 0; k <  poolLayers[numCLayers-1].outputDim_x; k++){
                (*delta_in3D)(k,j,i) = (*delta_in)(count++);
            }
        }
    }
//       delta_in3D->save("delta_in3D_initial.dat",arma::raw_ascii);
    for (int i = numCLayers - 1; i >= 0 ; i--){
        poolLayers[i].upSampling(delta_in3D);
        delta_in3D = poolLayers[i].delta_out;
//        delta_in3D->save("delta_in3D.dat",arma::raw_ascii);
        convoLayers[i].updatePara(delta_in3D, learningRate);
        delta_in3D = convoLayers[i].delta_out;
    }
}