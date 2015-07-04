#include "RBM.h"

using namespace NeuralNet;

RBM::RBM(int visibleDim0, int hiddenDim0, RBM::PreTrainPara preTrainPara0) {

    inputDim = visibleDim0;
    outputDim = hiddenDim0;
    trainingPara = preTrainPara0;
    H = std::make_shared<arma::umat>();
    V_reconstruct = std::make_shared<arma::umat>();
    H_reconstructProb = std::make_shared<arma::mat>();
    grad_B = std::make_shared<arma::vec>();
    grad_W = std::make_shared<arma::mat>();
    grad_B_old = std::make_shared<arma::vec>();
    grad_W_old = std::make_shared<arma::mat>();
    grad_A = std::make_shared<arma::vec>();
    grad_A_old = std::make_shared<arma::vec>();
    outputY = std::make_shared<arma::mat>();
    initializeWeight();
    W->save("initialWeight.dat",arma::raw_ascii);
    
    if (trainingPara.dropOutFlag) {
        randomGen = new Random_Bernoulli<unsigned long long>(trainingPara.dropOutRate);
    }
    
}

RBM::RBM(int visibleDim0, int hiddenDim0, std::shared_ptr<arma::umat> inputX0,
         PreTrainPara preTrainPara0):RBM(visibleDim0,hiddenDim0,preTrainPara0) {
    V = inputX0;
    numInstance = V->n_cols;
}

void RBM::initializeWeight() {
    W = std::make_shared<arma::mat>(outputDim,inputDim,arma::fill::randu);
    B = std::make_shared<arma::vec>(outputDim,arma::fill::randu);
    A = std::make_shared<arma::vec>(inputDim,arma::fill::randu);
    (*W) -= 0.5;
    (*B) -= 0.5;
    (*A) -= 0.5;
    (*W) *=4*sqrt(6.0/(inputDim+outputDim));
    (*B) *=4*sqrt(6.0/(inputDim+outputDim));
    (*A) *=4*sqrt(6.0/(inputDim+outputDim));
}


void RBM::train() {

    double energy;
    double energyTotal = 0;
    double errorTotal = 0;
    int ntimes = numInstance / trainingPara.miniBatchSize;
    int size = trainingPara.miniBatchSize;
    double learningRate = trainingPara.alpha / trainingPara.miniBatchSize;
    std::shared_ptr<arma::umat> subInputX;
    grad_W_old->zeros(outputDim,inputDim);
    grad_A_old->zeros(inputDim);
    grad_B_old->zeros(outputDim);
    int learningRateCount = 0;
/*    
    for (int i = 0; i < ntimes ; i++) {

        subInputX = std::make_shared<arma::umat>(V->cols(i*size,(i+1)*size-1));
        propUp(subInputX);
        reconstructVisible();
        reconstructHiddenProb();
        energyTotal +=calEnergy(subInputX);
    }
    std::cout << "energy is: " <<  energyTotal << std::endl;
*/
    for (int epoch = 0; epoch < trainingPara.NEpoch; epoch++) {
       
        if( ((learningRateCount+1)%10) == 0){
            learningRate *= trainingPara.learningRateDecay;
        }
        learningRateCount++;
        energyTotal = 0.0;
        errorTotal = 0.0;
        if( (epoch+1) % trainingPara.saveFrequency == 0){
            char tag[10];            
            sprintf(tag,"%d",(epoch+1));
            this->saveTrainResult("middleResult" + (std::string)tag);
        }
        
        for (int i = 0; i < ntimes ; i++) {
            subInputX = std::make_shared<arma::umat>(V->cols(i*size,(i+1)*size-1));
//  first is the sampling process
            propUp(subInputX);
            reconstructVisible();
            reconstructHiddenProb();
//        H->save("H.dat",arma::raw_ascii);
//        H_reconstructProb->save("H_reconstructProb.dat",arma::raw_ascii);
//        V_reconstruct->save("V_reconstruct.dat",arma::raw_ascii);

            *grad_W = ((*H_reconstructProb) * arma::conv_to<arma::mat>::from((*V_reconstruct).st())
                              - arma::conv_to<arma::mat>::from((*H)) * arma::conv_to<arma::mat>::from((*subInputX).st()));
//       grad.save("grad.dat",arma::raw_ascii);
            *grad_W += trainingPara.L2Decay * (*W);
            arma::mat gradBtemp = (*H_reconstructProb)- (*H);
            arma::mat gradAtemp = arma::conv_to<arma::mat>::from((*V_reconstruct))
                                  - arma::conv_to<arma::mat>::from((*subInputX));
//       arma::mat gradAtemp = arma::conv_to<arma::mat>::from(gradAtemp2);
            *grad_A = arma::sum(gradAtemp,1);
            *grad_B = arma::sum(gradBtemp,1);
            //      gradBtemp.save("gradBtemp.dat",arma::raw_ascii);
            //      gradAtemp.save("gradAtemp.dat",arma::raw_ascii);
//       gradAtemp2.save("gradAtemp2.dat",arma::raw_ascii);
//       gradA.print();
//       gradB.print();
            *grad_W_old = trainingPara.momentum * (*grad_W_old) - learningRate * (*grad_W);
            *grad_A_old = trainingPara.momentum * (*grad_A_old) - learningRate * (*grad_A);
            *grad_B_old = trainingPara.momentum * (*grad_B_old) - learningRate * (*grad_B);
            
            (*W) += (*grad_W_old);
            *A += (*grad_A_old);
            *B += (*grad_B_old);
            
//  here is the reconstruction error
            errorTotal += arma::as_scalar(grad_A->st() * (*grad_A));
//        energyTotal += calEnergy(subInputX);
        }
//        energy = calEnergy();
//        std::cout << "energy is: " <<  energyTotal << std::endl;
        std::cout << "epoch: " << epoch << "\t";
        std::cout << "learningRate: " << learningRate*trainingPara.miniBatchSize << "\t";
        std::cout << "gradient Norm is: " << arma::norm(*grad_W_old) << "\t";
        std::cout << "reconstruct error is: " << errorTotal << std::endl;
    }

//    this->saveTrainResult("finalTrain_");
}


void RBM::propUp(std::shared_ptr<arma::umat> subV) {

    (*outputY) = (*W) * (*subV);
    outputY->each_col() += (*B);
    outputY->transform([](double val) {
        return 1.0/(1+exp(-val));
    });
    arma::mat RandomMat = arma::randu(outputDim, outputY->n_cols);
    (*H) = RandomMat < (*outputY);

    if (trainingPara.dropOutFlag) {
        randomGen->modifier(H->memptr(), H->n_elem);   
    }
}

void RBM::reconstructVisible() {

    arma::mat Vtemp;
    Vtemp = (*W).st() * (*H);
    Vtemp.each_col() += (*A);
    Vtemp.transform([](double val) {
        return 1.0/(1+exp(-val));
    });
    arma::mat RandomMat = arma::randu(inputDim, Vtemp.n_cols);
    (*V_reconstruct) = RandomMat < Vtemp;
}

void RBM::reconstructHiddenProb() {

    (*H_reconstructProb) = (*W) * (*V_reconstruct);
    H_reconstructProb->each_col() += (*B);
    H_reconstructProb->transform([](double val) {
        return 1.0/(1+exp(-val));
    });

}

double RBM::calEnergy(std::shared_ptr<arma::umat> inputX) const {
    arma::vec en;
    double entemp;
//   A->print();
//   B->print();
//   H->print();
    en = (*inputX) * (*A) + (*H) * (*B);
//    en.print();
    en += arma::sum((*H) * (*W) * (*inputX).st(),1);
    entemp = -arma::sum(en);
//   std::cout << entemp << std::endl;
    return entemp;
}

void RBM::saveTrainResult(std::string filename) {

    W->save(filename+"W",arma::raw_ascii);
    A->save(filename+"A",arma::raw_ascii);
    B->save(filename+"B",arma::raw_ascii);

}

void RBM::loadTrainResult(std::string filename) {
    W->load(filename+"W",arma::raw_ascii);
    A->load(filename+"A",arma::raw_ascii);
    B->load(filename+"B",arma::raw_ascii);

}

double RBM::calReconstructError(std::shared_ptr<arma::umat> inputX) {

}


void RBM::TestViaReconstruct(std::shared_ptr<arma::mat> testDataX) {
    int numTest = testDataX->n_cols;

    std::shared_ptr<arma::umat> Vtest(new arma::umat(inputDim,numTest));
//    std::shared_ptr<arma::umat> Htest(new arma::umat(outputDim,numTest));
//    std::shared_ptr<arma::mat> testOutputY(new arma::mat(outputDim,numTest));
//    std::shared_ptr<arma::umat> Vtest_reconstruct(new arma::umat(inputDim, numTest));

    (*Vtest) = (*testDataX) > 0.5;
    
    // if we use drop out, we should scale the W when testing
    if(trainingPara.dropOutFlag) {
        (*W) *=trainingPara.dropOutRate;
    }
    propUp(Vtest);
    reconstructVisible();
    // if we use drop out, we should scale back the W after testing
    if(trainingPara.dropOutFlag) {
        (*W) /=trainingPara.dropOutRate;
    }
    
    arma::umat vtemp = arma::trans(*V_reconstruct);
    vtemp.save("test_reconstruct.dat",arma::raw_ascii);
}

void RBM::PreTrainPara::print() const {
    std::cout << eps << "\t";
    std::cout << NEpoch << "\t";
    std::cout << miniBatchSize << "\t";
    std::cout << alpha << "\t";
    std::cout << momentum << "\t";
    std::cout << learningRateDecay << "\t";
    std::cout << saveFrequency << "\t";
    std::cout << dropOutFlag << "\t";
    std::cout << dropOutRate << "\t";
    std::cout << std::endl;

}