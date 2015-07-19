#include "device_common.h"
#include "GPUMat.h"
#include "gtest/gtest.h"

TEST(GPUMATTest, selfAdd){
	GPUMat g1(5,5);
	GPUMat g2(5,5);
	
}

int main(){

	GPUEnv::GetInstance();

	GPUMat g(5,5);

	g.ones();
	g.print();


	return 0;

}
