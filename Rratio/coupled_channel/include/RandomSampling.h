#ifndef RANDOMSAMPLING_H 
#define RANDOMSAMPLING_H

#include "TVectorD.h"
#include "TMatrixDSym.h"
#include "TRandom3.h"


class RandomSampling {
    
public:
    
    RandomSampling( const TVectorD& vec, const TMatrixDSym& cov, const TMatrixDSym& inv );
    
    RandomSampling( const TVectorD& vec, const TMatrixDSym& cov );
    
    RandomSampling( const RandomSampling& other );
    
    ~RandomSampling() ;
    
    void random( std::vector<double>& par );

    TVectorD parameters() const ;
    
    TVectorD random() ;
    
    TVectorD uncorrelated() ;
    
    void seed( const UInt_t num ) ;
    
protected:
    
    TVectorD    vec_ ;
    TMatrixDSym cov_ ;
    TMatrixDSym inv_ ;
    
};

#endif
