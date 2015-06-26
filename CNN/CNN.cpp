
#include "CNN.h"


CNN::CNN(std::shared_ptr<arma::cube> trainingX0, std::shared_ptr<arma::mat> trainingY0, int nChanel0, TrainingPara trainingPara0){
    trainingX = trainingX0;
    trainingY = trainingY0;
    nChanel = nChanel0;
    numInstance = trainingX->n_slices / nChanel;
    inputDim_x = trainingX->n_rows;
    inputDim_y = trainingX->n_cols;
    outputDim = trainingY->n_rows;
    trainingPara = trainingPara0;
    
    numCLayers = 3;
    numFCLayers = 1;
    
    
    testGrad = false;
    totalDim = 0;
    
/* here is for reference for cifar 10 data:
 layer_defs = [];
layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3});
layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:2, stride:2});
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:2, stride:2});
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:2, stride:2});
layer_defs.push({type:'softmax', num_classes:10});
 */    
    
    convoLayers.push_back(ConvolveLayer(16,5,5,1, ConvolveLayer::ReLU));
//        convoLayers.push_back(ConvolveLayer(5,5,5,1, ConvolveLayer::ReLU));
//    convoLayers.push_back(ConvolveLayer(5,5,5,1, ConvolveLayer::ReLU)); 
    convoLayers.push_back(ConvolveLayer(20,5,5,1, ConvolveLayer::ReLU));
     convoLayers.push_back(ConvolveLayer(20,5,5,1, ConvolveLayer::ReLU));
//    poolLayers.push_back(PoolLayer(2,2,PoolLayer::max));
    poolLayers.push_back(PoolLayer(2,2,PoolLayer::max));;
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
        totalDim += convoLayers[0].totalSize;
    }
    
    
    FCLayers.push_back(BaseLayer(poolLayers[numCLayers-1].outputSize, outputDim, BaseLayer::softmax));
//    FCLayers.push_back(BaseLayer(100,outputDim,BaseLayer::sigmoid));
    totalDim += FCLayers[0].totalSize;
    
    if(trainingPara.loadFlag) loadWeight();
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
    double error, errorTotal, crossEntropy;
    int size = trainingPara.miniBatchSize;
    for (int epoch = 0; epoch < trainingPara.NEpoch; epoch++) {
        std::cout << epoch << std::endl;
        ntimes  = numInstance / trainingPara.miniBatchSize;
        errorTotal = 0.0;
        crossEntropy = 0.0;
        if(((epoch + 1)%trainingPara.saveFrequency) == 0) saveWeight();
        for (int i = 0; i < ntimes; i++) {
//            std::cout << i << "minibatch in " << ntimes << std::endl;
            (*subInput) = trainingX->slices(i*size*nChanel,(i+1)*size*nChanel-1);
            (*subInputY) = trainingY->cols(i*size,(i+1)*size-1);   
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
 //           output->print();
//            subInputY->print("Y");
            (*delta) = (*output) - (*subInputY);
            
            
//            subInputY->print();
            backProp(delta);     
            
            delta->transform([](double val){return val*val;});
            error = arma::sum(arma::sum((*delta)));
            errorTotal += error;  
           output->transform([](double val){return std::log(val+1e-20);});
           arma::mat crossEntropy_temp = *(subInputY) %  *(output);
//    crossEntropy_temp.save("crossEntropy.dat",arma::raw_ascii);
            crossEntropy -= arma::sum(arma::sum((crossEntropy_temp)));
                       
        }
        std::cout << "error is: " << errorTotal << std::endl;
         std::cout << "cross entropy is: " << crossEntropy << std::endl;
    }
}

void CNN::calNumericGrad(std::shared_ptr<arma::cube> subInput,std::shared_ptr<arma::mat> subInputY){
    std::shared_ptr<arma::mat> delta = std::make_shared<arma::mat>();
    int dim1 = convoLayers[0].numFilters;
    int dim2 = convoLayers[0].inputDim_z;
    int matDim1 = convoLayers[0].filterDim_x;
    int matDim2 = convoLayers[0].filterDim_y;
    double eps = 1e-7;
       
    Tensor_4D::ptr dW= Tensor_4D::build(matDim1, matDim2, dim1, dim2);
    
    int ntimes;
    double error, errorTotal;
    double temp_left, temp_right;

    
    for (int i = 0; i < dim1; i++){
        for (int j = 0; j < dim2; j++){
            for (int m = 0; m < matDim1; m++){
                for (int n = 0; n < matDim2; n++){
                    (*(convoLayers[0].filters))(n,m,j,i) += eps;
                    feedForward(subInput);
                    (*delta) = ((*output) - (*subInputY)).st();
                    *delta = arma::sum(*delta,1);
                    error = 0.5* arma::as_scalar((*delta).st() * (*delta));
                    temp_left = error;
                    (*(convoLayers[0].filters))(n,m,j,i) -= 2.0*eps;
                    feedForward(subInput);
                    (*delta) = ((*output) - (*subInputY)).st();
                    *delta = arma::sum(*delta,1);
                    error = 0.5* arma::as_scalar((*delta).st() * (*delta));
                    temp_right = error;
                    (*(convoLayers[0].filters))(n,m,j,i) += eps;
                    
                    (*dW)(m,n,i,j) = (temp_left - temp_right) / 2.0 / eps;
                }           
            }       
        }         
    }  
//    MatArray<double>::save(dW,"numGrad_Conv");
    
    
    
    
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
    int inputInstance = subInput->n_slices / nChanel;
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
    std::shared_ptr<arma::mat> subInput_mat = std::make_shared<arma::mat>(subInput->memptr(),totalSize, inputInstance);
 /*   
    for (int instance = 0; instance < inputInstance; instance++){
    int count = 0;    
    for (int i = 0; i <  poolLayers[numCLayers-1].outputDim_z; i++){
        for (int j = 0; j <  poolLayers[numCLayers-1].outputDim_x; j++){
            for (int k = 0; k <  poolLayers[numCLayers-1].outputDim_y; k++){
                (*subInput_mat)(instance,count++) = (*subInput)(k, j ,i + instance * poolLayers[numCLayers-1].outputDim_z);
            }
        }
    }
    }
  */ 
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
    int inputInstance = delta_in->n_cols;
//    delta_target->save("delta_target.dat",arma::raw_ascii);
    double learningRate = trainingPara.alpha / trainingPara.miniBatchSize;
    for (int i = numFCLayers-1; i >= 0 ; i--){
        FCLayers[i].updatePara(delta_in, learningRate);
        delta_in = FCLayers[i].delta_out;        
    }
//  remember to transform to 3d    ,
//  here transform 1d to 3d    
//    delta_in->save("FCLayer_deltaout.dat",arma::raw_ascii);
    std::shared_ptr<arma::cube> delta_in3D = std::make_shared<arma::cube>(delta_in->memptr(),poolLayers[numCLayers-1].outputDim_x,
            poolLayers[numCLayers-1].outputDim_y, poolLayers[numCLayers-1].outputDim_z * inputInstance);
/*
    for (int instance = 0; instance < inputInstance; instance++){
        int count = 0;
        for (int i = 0; i <  poolLayers[numCLayers-1].outputDim_z; i++){
        for (int j = 0; j <  poolLayers[numCLayers-1].outputDim_x; j++){
            for (int k = 0; k <  poolLayers[numCLayers-1].outputDim_y; k++){
                (*delta_in3D)(k,j,i + instance*poolLayers[numCLayers-1].outputDim_z) = (*delta_in)(count++, instance);
            }
        }
    }
    }
 */
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
//    int size = trainingPara.miniBatchSize;
//        std::cout << "test result" << std::endl;
//        ntimes  = testX0->n_slices / nChanel / trainingPara.miniBatchSize;
        errorTotal = 0.0;
//        for (int i = 0; i < ntimes; i++) {
            subInput = testX0;
            subInputY = testY0;            
            feedForward(subInput);
//            output->print();
            (*delta) = ((*output) - (*subInputY));
            delta->transform([](double val){return val*val;});
            error = arma::sum(arma::sum((*delta)));        
            errorTotal += error;                        
//        }
            output->save("test_output.dat",arma::raw_ascii);
            subInputY->save("test_label.dat",arma::raw_ascii);
            delta->save("test_delta.dat",arma::raw_ascii);
        std::cout << "test result" << std::endl;
        std::cout << errorTotal << std::endl;
    
}

/*
double CNN::calLayerError(std::shared_ptr<arma::cube> delta){
    double total = 0.0;
    for (int i = 0; i < delta->n_slices; i++)
        total += arma::sum(arma::sum(delta->slice(i)));
    return total;
}
*/

void CNN::calGrad(std::shared_ptr<arma::mat> delta_target){
    std::shared_ptr<arma::mat> delta_in = delta_target;
    int inputInstance = delta_in->n_cols;
//    delta_target->save("delta_target.dat",arma::raw_ascii);
    double learningRate = trainingPara.alpha / trainingPara.miniBatchSize;
    for (int i = numFCLayers-1; i >= 0 ; i--){
        FCLayers[i].calGrad(delta_in);
        delta_in = FCLayers[i].delta_out;        
    }
//  remember to transform to 3d    ,
//  here transform 1d to 3d    
//    delta_in->save("FCLayer_deltaout.dat",arma::raw_ascii);
    std::shared_ptr<arma::cube> delta_in3D = std::make_shared<arma::cube>(delta_in->memptr(),poolLayers[numCLayers-1].outputDim_x,
            poolLayers[numCLayers-1].outputDim_y, poolLayers[numCLayers-1].outputDim_z * inputInstance);

//       delta_in3D->save("delta_in3D_initial.dat",arma::raw_ascii);
    
    for (int i = numCLayers - 1; i >= 0 ; i--){
        poolLayers[i].upSampling(delta_in3D);
        delta_in3D = poolLayers[i].delta_out;
//        delta_in3D->save("Pool_delta_in3D.dat",arma::raw_ascii);
        convoLayers[i].calGrad_matrixMethod(delta_in3D);
        delta_in3D = convoLayers[i].delta_out;
    }
}


void CNN::deVectoriseWeight(arma::vec &x){
    int startIdx = 0;
    int endIdx = 0;
//        std::shared_ptr<arma::vec> V(new arma::vec);
    for (int i = 0; i < numCLayers; i++) {
        startIdx = endIdx;
        endIdx += convoLayers[i].totalSize;
 //       *V = x.rows(startIdx,endIdx - 1);   
        convoLayers[i].deVectoriseWeight(x.memptr(),startIdx);        
    }
    
    
    for (int i = 0; i < numFCLayers; i++) {
        startIdx = endIdx;
        endIdx += FCLayers[i].totalSize;
 //       *V = x.rows(startIdx,endIdx - 1);   
        FCLayers[i].deVectoriseWeight(x.memptr(),startIdx);        
    }
}

void CNN::vectoriseGrad(arma::vec &grad){
        int startIdx = 0;
        int endIdx = 0;
//        std::shared_ptr<arma::vec> V(new arma::vec);
    for (int i = 0; i < numCLayers; i++) {
        startIdx = endIdx;
        endIdx += convoLayers[i].totalSize;
 
        convoLayers[i].vectoriseGrad(grad.memptr(), startIdx);
//        V->print();
//        grad.rows(startIdx,endIdx - 1) = *V;   
    }
        
    for (int i = 0; i < numFCLayers; i++) {
        startIdx = endIdx;
        endIdx += FCLayers[i].totalSize;
 
        FCLayers[i].vectoriseGrad(grad.memptr(), startIdx);
//        V->print();
//        grad.rows(startIdx,endIdx - 1) = *V;   
    }   
        
        
        
}

void CNN::vectoriseWeight(arma::vec &x){
    int startIdx = 0;
    int endIdx = 0;
//        std::shared_ptr<arma::vec> V(new arma::vec);
    for (int i = 0; i < numCLayers; i++) {
        startIdx = endIdx;
        endIdx += convoLayers[i].totalSize;
        
        convoLayers[i].vectoriseWeight(x.memptr(), startIdx);
//        x.rows(startIdx,endIdx - 1) = *V;
    }
    
    for (int i = 0; i < numFCLayers; i++) {
        startIdx = endIdx;
        endIdx += FCLayers[i].totalSize;
        
        FCLayers[i].vectoriseWeight(x.memptr(), startIdx);
//        x.rows(startIdx,endIdx - 1) = *V;
    }
}

void CNN::loadWeight(std::string str){
    arma::vec x(totalDim);
    x.load(str,arma::raw_ascii);
    deVectoriseWeight(x);
}
void CNN::saveWeight(std::string str){
    arma::vec x(totalDim);
    
    vectoriseWeight(x);    
    x.save(str,arma::raw_ascii);
}

CNNTrainer::CNNTrainer(CNN& CNN0):cnn(CNN0){
    dim = cnn.totalDim;  
    x_init = std::make_shared<arma::vec>(dim);
    cnn.vectoriseWeight(*x_init);
//    x_init->save("x_init.dat", arma::raw_ascii);
}

void CNNTrainer::gradientChecking(){

    arma::vec grad_ana;
    this->operator()(*(this->x_init), grad_ana);

    grad_ana.save("grad_ana.dat",arma::raw_ascii);
    int ncheck = 20;
    arma::vec graddummy;
    arma::vec grad_num(ncheck,2);
    double eps = 1e-7;
   // randomly pick 20 to compare 
    for (int i = 0; i < ncheck; i++){
        
        int index = dim * arma::randu();
        
        (this->x_init)->at(index) += eps ;
        double left = this->operator()(*(this->x_init), graddummy);
        (this->x_init)->at(index) -= 2.0*eps ;
        double right = this->operator()(*(this->x_init), graddummy);
        (this->x_init)->at(index) += eps ;
        grad_num(i,0) = index;
        grad_num(i,1) = (left - right)/2.0 / eps;
        
        std::cout << "check: " << i <<" : "<< index << std::endl;
        std::cout << grad_ana(index) << std::endl;
        std::cout << grad_num(i,1) << std::endl;
    }
    
    grad_num.save("grad_num.dat",arma::raw_ascii);


}
double CNNTrainer::operator ()(arma::vec& x, arma::vec& grad){

    grad.resize(cnn.totalDim);
//  first assign x to the weights and bias of all the layers    
    cnn.deVectoriseWeight(x);
    
    cnn.feedForward(cnn.trainingX);
    std::shared_ptr<arma::mat> delta(new arma::mat);
    //for delta: each column is the delta of a sample
    *delta = (-*(cnn.trainingY) + *(cnn.output));            
//    arma::vec error = arma::sum(*delta,1);
//  the error function we should have is the cross entropy 
//  since our gradient calcuated is assuming that we are using cross entropy
//    error.save("error.dat",arma::raw_ascii);
     cnn.output->transform([](double val){return std::log(val+1e-20);});

    arma::mat crossEntropy_temp = *(cnn.trainingY) %  *(cnn.output);
//    crossEntropy_temp.save("crossEntropy.dat",arma::raw_ascii);
    double crossEntropy = - arma::sum(arma::sum((crossEntropy_temp)));
//     std::cout << "cross entropy is: " << crossEntropy << std::endl;
    double errorTotal= 0.5 * arma::sum(arma::sum(delta->st() * (*delta)));            
    cnn.calGrad(delta);
    
    cnn.vectoriseGrad(grad);
    return crossEntropy;
}



