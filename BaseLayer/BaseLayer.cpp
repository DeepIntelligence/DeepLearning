#include <cmath>
#include "BaseLayer.h"

BaseLayer::BaseLayer(int inputDim0, int outputDim0, ActivationType actType0){
  inputDim = inputDim0;
  outputDim = outputDim0;
  actType = actType0;
  initializeWeight();  
};

void BaseLayer::initializeWeight(){
    W.randn(outputDim,inputDim);
    B.randn(outputDim);
    W -= 0.5;
    B -= 0.5;

    if (actType == sigmoid){
    W *=4*sqrt(6.0/(inputDim+outputDim));
    B *=4*sqrt(6.0/(inputDim+outputDim));
    } else if (actType == softmax){
    W *=sqrt(6.0/(inputDim+outputDim));
    B *=sqrt(6.0/(inputDim+outputDim));    
       
    }

}

void BaseLayer::save(std::string filename){
    W.save(filename+"_W.dat",arma::raw_ascii);
    B.save(filename+"_B.dat",arma::raw_ascii);

}

void BaseLayer::activateUp(){
  outputY = std::make_shared<arma::mat>(numInstance,outputDim);
  std::shared_ptr<arma::mat> &p=outputY;
// first get the projection  
  (*p) = (*inputX) * W.st() ;
  
  for (int i = 0; i < inputX->n_rows; i++) p->row(i) += B.st();  
// then do the activation  
  arma::mat maxVal = arma::max(*p,1);
  switch(actType){
    case softmax:
//        p->print();
//        maxVal.print();
    for (int i = 0; i < inputX->n_rows; i++){
      for (int j = 0; j < outputDim; j++){
        (*p)(i,j)-= maxVal(i);        
      }
    }    
//        p->print();
    (*p).transform([](double val){return exp(val); }) ;
//    p->print();
    double normfactor;
    for (int i = 0; i < inputX->n_rows; i++){
      normfactor = 0.0;
      for (int j = 0; j < outputDim; j++){
        normfactor+=p->at(i,j);        
      }
      for (int j = 0; j < outputDim; j++){
        (*p)(i,j)/=normfactor;        
      }
    }    
//    std::cout << normfactor << std::endl;
//    p->print();
    break;
    case sigmoid:
//        p->print();
    (*p).transform([](double val){return 1.0/(1.0+exp(-val)); }) ; 
    break;
    case linear:
    break; 
  }
}
  
      
    	
	
	
	
