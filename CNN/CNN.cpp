
#include "CNN.h"


CNN::CNN(std::shared_ptr<arma::cube> trainingX0, std::shared_ptr<arma::mat> trainingY0, int nChanel0, TrainingPara trainingPara0){
    trainingX = trainingX0;
    trainingY = trainingY0;
    nChanel = nChanel0;
    numInstance = trainingX->n_slices / nChanel;
    inputDim_x = trainingX->n_rows;
    inputDim_y = trainingX->n_cols;
    outputDim = trainingY->n_cols;
    trainingPara = trainingPara0;
    
    numCLayers = 3;
    numFCLayers = 1;
    
    
    testGrad = false;
    
    
    convoLayers.push_back(ConvolveLayer(16,5,5,1, ConvolveLayer::ReLU));
    convoLayers.push_back(ConvolveLayer(20,5,5,1, ConvolveLayer::ReLU)); 
    convoLayers.push_back(ConvolveLayer(20,5,5,1, ConvolveLayer::ReLU));
    poolLayers.push_back(PoolLayer(2,2,PoolLayer::max));
//    poolLayers.push_back(PoolLayer(2,2,PoolLayer::max));
//    convoLayers.push_back(ConvolveLayer(10,5,5,1, ConvolveLayer::tanh));
    poolLayers.push_back(PoolLayer(2,2,PoolLayer::max));
    poolLayers.push_back(PoolLayer(2,2,PoolLayer::max));
    
    for (int i = 0; i < numCLayers; i++){
        if (i == 0)
            convoLayers[0].setInputDim(inputDim_x, inputDim_y, nChanel);
        else
            convoLayers[i].setInputDim(poolLayers[i-1].outputDim_x,
                              poolLayers[i-1].outputDim_y,
                              poolLayers[i-1].outputDim_z);
        
        poolLayers[i].setInputDim(convoLayers[i].outputDim_x,
                              convoLayers[i].outputDim_y,
                              convoLayers[i].outputDim_z);
    }
    
    
    FCLayers.push_back(BaseLayer(poolLayers[numCLayers-1].outputSize, outputDim, BaseLayer::sigmoid));
//    FCLayers.push_back(BaseLayer(100,outputDim,BaseLayer::sigmoid));

}

void CNN::setTrainingData(std::shared_ptr<arma::cube> trainingX0, std::shared_ptr<arma::mat> trainingY0, int nChanel0){
    trainingX = trainingX0;
    trainingY = trainingY0;
    nChanel = nChanel0;
    numInstance = trainingX->n_slices / nChanel;
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
//            subInput->ones();
//            FCLayers[0].W->eye();
//            FCLayers[0].B->zeros();
//            FCLayers[1].W->ones();
//            FCLayers[1].B->zeros();
//            (*convoLayers[0].filters)[0][0].ones();
//            (*convoLayers[0].B).zeros();
//            (*convoLayers[1].filters)[0][0].ones();
//            (*convoLayers[1].B).zeros();

            feedForward(subInput);
            if (testGrad){
                calNumericGrad(subInput, subInputY);
                feedForward(subInput);
            }
//            output->print();
            (*delta) = ((*output) - (*subInputY)).st();
//            subInputY->print();
            backProp(delta);            
            error = arma::sum(arma::sum((*delta).st() * (*delta)));
            errorTotal += error;  
        }
        std::cout << errorTotal << std::endl;
    }
}

void CNN::calNumericGrad(std::shared_ptr<arma::cube> subInput,std::shared_ptr<arma::mat> subInputY){
    std::shared_ptr<arma::mat> delta = std::make_shared<arma::mat>();
    int dim1 = convoLayers[0].numFilters;
    int dim2 = convoLayers[0].inputDim_z;
    int matDim1 = convoLayers[0].filterDim_x;
    int matDim2 = convoLayers[0].filterDim_y;
    double eps = 1e-7;
       
    MatArray<double>::Mat2DArray_ptr dW= MatArray<double>::build( dim1, dim2,  matDim1, matDim2);
    
    int ntimes;
    double error, errorTotal;
    double temp_left, temp_right;

    
    for (int i = 0; i < dim1; i++){
        for (int j = 0; j < dim2; j++){
            for (int m = 0; m < matDim1; m++){
                for (int n = 0; n < matDim2; n++){
                    (*(convoLayers[0].filters))[i][j](m,n) += eps;
                    feedForward(subInput);
                    (*delta) = ((*output) - (*subInputY)).st();
                    *delta = arma::sum(*delta,1);
                    error = 0.5* arma::as_scalar((*delta).st() * (*delta));
                    temp_left = error;
                    (*(convoLayers[0].filters))[i][j](m,n) -= 2.0*eps;
                    feedForward(subInput);
                    (*delta) = ((*output) - (*subInputY)).st();
                    *delta = arma::sum(*delta,1);
                    error = 0.5* arma::as_scalar((*delta).st() * (*delta));
                    temp_right = error;
                    (*(convoLayers[0].filters))[i][j](m,n) += eps;
                    
                    (*dW)[i][j](m,n) = (temp_left - temp_right) / 2.0 / eps;
                }           
            }       
        }         
    }  
    MatArray<double>::save(dW,"numGrad_Conv");
    
    
    
    
    std::shared_ptr<arma::mat> delta2 = std::make_shared<arma::mat>();
    dim1 = FCLayers[0].outputDim;
    dim2 = FCLayers[0].inputDim;
//    double eps = 1e-9;
       
    arma::mat dW2(dim1,dim2,arma::fill::zeros);

    
    for (int i = 0; i < dim1; i++){
        for (int j = 0; j < dim2; j++){
            (*(FCLayers[0].W))(i,j) += eps;
            feedForward(subInput);
 //           outputY->transform([](double val){return log(val);});
            (*delta2) = ((*output) - (*subInputY)).st();
            *delta2 = arma::sum(*delta2,1);
            error = 0.5* arma::as_scalar((*delta2).st() * (*delta2));
            temp_left = error;
            (*(FCLayers[0].W))(i,j) -= 2.0*eps;
            feedForward(subInput);
 //           outputY->transform([](double val){return log(val);});
            (*delta2) = ((*output) - (*subInputY)).st();
            *delta2 = arma::sum(*delta2,1);
            error = 0.5* arma::as_scalar((*delta2).st() * (*delta2));;
            temp_right = error;
            (*(FCLayers[0].W))(i,j) += eps;
            dW2(i,j) = (temp_left - temp_right) / 2.0 / eps;    
        }         
    }  
    dW2.save("numGrad_FCLayer.dat",arma::raw_ascii);
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

 // here is convert 3d cube to 1d vector   
    int totalSize = poolLayers[numCLayers-1].outputSize;   
 //   subInput->reshape(1,totalSize,1);
    std::shared_ptr<arma::mat> subInput_mat = std::make_shared<arma::mat>(1,totalSize);
    int count = 0;
    for (int i = 0; i <  poolLayers[numCLayers-1].outputDim_z; i++){
        for (int j = 0; j <  poolLayers[numCLayers-1].outputDim_x; j++){
            for (int k = 0; k <  poolLayers[numCLayers-1].outputDim_y; k++){
                (*subInput_mat)(0,count++) = (*subInput)(j,k,i);
            }
        }
    }
//    subInput_mat->save("subinputmat.dat",arma::raw_ascii);
    for (int i = 0; i < numFCLayers; i++) {
        FCLayers[i].activateUp(subInput_mat);
        subInput_mat = FCLayers[i].outputY;
    }
    output = FCLayers[numFCLayers-1].outputY;

//    output->save("networkoutput.dat", arma::raw_ascii);
}

void CNN::backProp(std::shared_ptr<arma::mat> delta_target){
    std::shared_ptr<arma::mat> delta_in = delta_target;
//    delta_target->save("delta_target.dat",arma::raw_ascii);
    double learningRate = trainingPara.alpha / trainingPara.miniBatchSize;
    for (int i = numFCLayers-1; i >= 0 ; i--){
        FCLayers[i].updatePara(delta_in, learningRate);
        delta_in = FCLayers[i].delta_out;        
    }
//  remember to transform to 3d    ,
//  here transform 1d to 3d    
//    delta_in->save("FCLayer_deltaout.dat",arma::raw_ascii);
    std::shared_ptr<arma::cube> delta_in3D = std::make_shared<arma::cube>(poolLayers[numCLayers-1].outputDim_x,
            poolLayers[numCLayers-1].outputDim_y, poolLayers[numCLayers-1].outputDim_z);
    int count = 0;
    for (int i = 0; i <  poolLayers[numCLayers-1].outputDim_z; i++){
        for (int j = 0; j <  poolLayers[numCLayers-1].outputDim_x; j++){
            for (int k = 0; k <  poolLayers[numCLayers-1].outputDim_y; k++){
                (*delta_in3D)(j,k,i) = (*delta_in)(count++);
            }
        }
    }
//       delta_in3D->save("delta_in3D_initial.dat",arma::raw_ascii);
    
    for (int i = numCLayers - 1; i >= 0 ; i--){
        poolLayers[i].upSampling(delta_in3D);
        delta_in3D = poolLayers[i].delta_out;
//        delta_in3D->save("Pool_delta_in3D.dat",arma::raw_ascii);
        convoLayers[i].updatePara(delta_in3D, learningRate);
        delta_in3D = convoLayers[i].delta_out;
    }
}

void CNN::test(std::shared_ptr<arma::cube> testX0, std::shared_ptr<arma::mat> testY0) {
    std::shared_ptr<arma::cube> subInput = std::make_shared<arma::cube>();
    std::shared_ptr<arma::mat> subInputY = std::make_shared<arma::mat>();
    std::shared_ptr<arma::mat> delta = std::make_shared<arma::mat>();
    int ntimes;
    double error, errorTotal;
    int size = trainingPara.miniBatchSize;
        std::cout << "test result" << std::endl;
        ntimes  = testX0->n_slices / nChanel / trainingPara.miniBatchSize;
        errorTotal = 0.0;
        for (int i = 0; i < ntimes; i++) {
            (*subInput) = testX0->slices(i*size*nChanel,(i+1)*size*nChanel-1);
            (*subInputY) = testY0->rows(i*size,(i+1)*size-1);            
            feedForward(subInput);
//            output->print();
            (*delta) = ((*output) - (*subInputY)).st();
//            subInputY->print();          
            error = arma::sum(arma::sum((*delta).st() * (*delta)));
            errorTotal += error;                        
        }
        std::cout << errorTotal << std::endl;
    
}

double CNN::calLayerError(std::shared_ptr<arma::cube> delta){
    double total = 0.0;
    
    for (int i = 0; i < delta->n_slices; i++)
        total += arma::sum(arma::sum(delta->slice(i)));
    
    return total;


}