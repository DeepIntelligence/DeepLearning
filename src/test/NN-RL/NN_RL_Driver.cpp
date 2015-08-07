#include "model.h"
#include "Util.h"
void testModel();

int main(){

    testModel();
    

    return 0;
}


void testModel(){
    double dt = 0.05;
    Model model(dt);
    
    model.createInitialState();
    
    double T = 1000;
    double t = 0.0;
    while (t < T && !model.terminate()) {
        
        std::cout << t << "\t";
        std::cout << model.getCurrState().theta << "\t " << model.getCurrState().theta_v << std::endl;
        
        model.run(1);
    
        t += dt;
    }
    
    
    
    
    
    



}