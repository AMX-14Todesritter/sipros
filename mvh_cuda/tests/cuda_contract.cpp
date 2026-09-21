#include "engine.h"
#include "proNovoConfig.h"
#include <iostream>
int main(int argc,char **argv){
    try{
        if(argc!=2||!ProNovoConfig::setFilename(argv[1]))return 2;
        mvh_cuda::runContractTests();return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
