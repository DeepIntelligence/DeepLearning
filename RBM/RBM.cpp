#include "RBM.h"

RBM::RBM(int visibleDim0, int hiddenDim0, RBM::PreTrainPara preTrainPara0){
	
	inputDim = visibleDim0;
	outputDim = hiddenDim0;
	trainingPara = preTrainPara0;
        H = std::make_shared<arma::umat>();
	V_reconstruct = std::make_shared<arma::umat>();
        H_reconstructProb = std::make_shared<arma::mat>();
	initializeWeight();
        W->save("initialWeight.dat",arma::raw_ascii);

}

RBM::RBM(int visibleDim0, int hiddenDim0, std::shared_ptr<arma::umat> inputX0,
         PreTrainPara preTrainPara0):RBM(visibleDim0,hiddenDim0,preTrainPara0){
        V = inputX0;
        numInstance = V->n_rows;
}

void RBM::initializeWeight(){
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


void RBM::train(){
	
    double energy;
    double energyTotal = 0;
    double errorTotal = 0;
    int ntimes = numInstance / trainingPara.miniBatchSize;
    int size = trainingPara.miniBatchSize;
    std::shared_ptr<arma::umat> subInputX;
 
    for (int i = 0; i < ntimes ; i++){            
            subInputX = std::make_shared<arma::umat>(V->rows(i*size,(i+1)*size-1));
             propUp(subInputX);
            reconstructVisible();
            reconstructHiddenProb();
            energyTotal +=calEnergy(subInputX);
    }
        std::cout << "energy is: " <<  energyTotal << std::endl;
    
        
    for (int epoch = 0; epoch < trainingPara.NEpoch; epoch++){
        std::cout << "epoch: " << epoch << std::endl;         
        energyTotal = 0.0;
        errorTotal = 0.0;
        for (int i = 0; i < ntimes ; i++){            
            subInputX = std::make_shared<arma::umat>(V->rows(i*size,(i+1)*size-1));
//  first is the sampling process    
            propUp(subInputX);
            reconstructVisible();
            reconstructHiddenProb();
//        H->save("H.dat",arma::raw_ascii);
//        H_reconstructProb->save("H_reconstructProb.dat",arma::raw_ascii);
//        V_reconstruct->save("V_reconstruct.dat",arma::raw_ascii);
            
        arma::mat grad = ((*H_reconstructProb).st() * arma::conv_to<arma::mat>::from((*V_reconstruct))
                - arma::conv_to<arma::mat>::from((*H).st()) * arma::conv_to<arma::mat>::from((*subInputX)));
 //       grad.save("grad.dat",arma::raw_ascii);
        
        (*W) -= trainingPara.alpha * grad;
        arma::mat gradBtemp = (*H_reconstructProb).st()- (*H).st();
        arma::mat gradAtemp = arma::conv_to<arma::mat>::from((*V_reconstruct).st())
                - arma::conv_to<arma::mat>::from((*subInputX).st());
 //       arma::mat gradAtemp = arma::conv_to<arma::mat>::from(gradAtemp2);
        arma::vec gradA = arma::sum(gradAtemp,1);
        arma::vec gradB = arma::sum(gradBtemp,1);
  //      gradBtemp.save("gradBtemp.dat",arma::raw_ascii);
  //      gradAtemp.save("gradAtemp.dat",arma::raw_ascii);
 //       gradAtemp2.save("gradAtemp2.dat",arma::raw_ascii);
 //       gradA.print();
 //       gradB.print();
        *A -= trainingPara.alpha * (gradA);
        *B -= trainingPara.alpha * (gradB);                          

        errorTotal += arma::as_scalar(gradA.st() * gradA);
//        energyTotal += calEnergy(subInputX);
        }
//        energy = calEnergy();
//        std::cout << "energy is: " <<  energyTotal << std::endl;
        std::cout << "reconstruct error is: " << errorTotal << std::endl;
    }
	
    
    W->save("finalWeight.dat",arma::raw_ascii);
}


void RBM::propUp(std::shared_ptr<arma::umat> subV){
	
    outputY = std::make_shared<arma::mat>(subV->n_rows,outputDim);
    (*outputY) = (*subV) * (*W).st();
//    arma::mat outputY = (*V) * (*W).st();
    for (int i = 0; i < outputY->n_rows; i++) outputY->row(i) += (*B).st();  
    outputY->transform([](double val){return 1.0/(1+exp(-val));});	
    arma::mat RandomMat = arma::randu(outputY->n_rows, outputDim);
    (*H) = RandomMat < (*outputY);   
			
}

void RBM::reconstructVisible(){	
	
    arma::mat Vtemp;
    Vtemp = (*H) * (*W);
    for (int i = 0; i < Vtemp.n_rows; i++) Vtemp.row(i) += (*A).st();  
    Vtemp.transform([](double val){return 1.0/(1+exp(-val));});
    arma::mat RandomMat = arma::randu(Vtemp.n_rows,inputDim);
    (*V_reconstruct) = RandomMat < Vtemp;
}

void RBM::reconstructHiddenProb(){	
	
    (*H_reconstructProb) = (*V_reconstruct) * (*W).st();
    for (int i = 0; i < H_reconstructProb->n_rows; i++) H_reconstructProb->row(i) += (*B).st();  
    H_reconstructProb->transform([](double val){return 1.0/(1+exp(-val));});	

        
}

double RBM::calEnergy(std::shared_ptr<arma::umat> inputX) const{
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

void RBM::saveTrainResult(std::string filename){
    
    W->save(filename+"W",arma::raw_ascii);
    A->save(filename+"A",arma::raw_ascii);
    B->save(filename+"B",arma::raw_ascii);

}

void RBM::loadTrainResult(std::string filename){
    W->load(filename+"W",arma::raw_ascii);
    A->load(filename+"A",arma::raw_ascii);
    B->load(filename+"B",arma::raw_ascii);


}

double RBM::calReconstructError(std::shared_ptr<arma::umat> inputX){


}


void RBM::TestViaReconstruct(std::shared_ptr<arma::mat> testDataX){
    int numTest = testDataX->n_rows;
    
    std::shared_ptr<arma::umat> Vtest(new arma::umat(numTest,inputDim));
    std::shared_ptr<arma::umat> Htest(new arma::umat(numTest,outputDim));
    std::shared_ptr<arma::mat> testOutputY(new arma::mat(numTest,outputDim));
    std::shared_ptr<arma::umat> Vtest_reconstruct(new arma::umat(numTest,inputDim));
    
    (*Vtest) = (*testDataX) > 0.5;
    

    (*testOutputY) = (*Vtest) * (*W).st();
    for (int i = 0; i < numTest; i++) testOutputY->row(i) += (*B).st();  
    testOutputY->transform([](double val){return 1.0/(1+exp(-val));});	
    arma::mat RandomMat = arma::randu(numTest, outputDim);
    (*Htest) = RandomMat < (*testOutputY);  
    
    arma::mat Vtemp;
    Vtemp = (*Htest) * (*W);
    for (int i = 0; i < numTest; i++) Vtemp.row(i) += (*A).st();  
    Vtemp.transform([](double val){return 1.0/(1+exp(-val));});
    RandomMat = arma::randu(numTest,inputDim);
    (*Vtest_reconstruct) = RandomMat < Vtemp;

    Vtest_reconstruct->save("reconstruct.dat",arma::raw_ascii);
}

void RBM::PreTrainPara::print() const{
  std::cout << eps << "\t";
  std::cout << NEpoch << "\t";
  std::cout << miniBatchSize << "\t";
  std::cout << alpha << std::endl;

}