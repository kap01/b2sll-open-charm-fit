#include "RandomSampling.h" 

#include <cmath>
#include <cassert>
#include <iostream>

#include "TDecompSVD.h"
#include "TDecompChol.h"

#include "TRandom3.h"


RandomSampling::RandomSampling( const TVectorD& vec, const TMatrixDSym& cov ) :
    vec_ ( vec ),
    cov_ ( cov ),
    inv_ ( cov ){
    
    inv_.Invert();
} 

RandomSampling::RandomSampling( const TVectorD& vec, const TMatrixDSym& cov, const TMatrixDSym& inv ) :
    vec_ ( vec ),
    cov_ ( cov ),
    inv_ ( inv ){} 


RandomSampling::RandomSampling( const RandomSampling& other ) :
    vec_ ( other.vec_ ),
    cov_ ( other.cov_ ),
    inv_ ( other.inv_ ){} 


RandomSampling::~RandomSampling() { } 



void RandomSampling::seed( const UInt_t num ) {
    gRandom->SetSeed( num );
}

TVectorD RandomSampling::parameters() const {
    return vec_;
}


TVectorD RandomSampling::uncorrelated() {
  TVectorD par( vec_.GetNrows() );
  
  for ( int i = 0; i < vec_.GetNrows(); ++i ) {
    par(i) = gRandom->Gaus( vec_(i), std::sqrt( cov_(i,i) ) );
  }
  
  return par;
}

TVectorD RandomSampling::random(){
    
  const int nrows = vec_.GetNrows();
  
  if ( 1 == nrows ){ 
    return uncorrelated();
  }

  TVectorD par( nrows );
  
  TVectorD tmp( nrows );
  
  for ( int i = 0; i < nrows; ++i ){
    tmp(i) = gRandom->Gaus( 0., 1. );
  }
  
  TDecompChol cholesky( cov_ );
  bool success = cholesky.Decompose();
  
  assert(success);
  
  TMatrixD U  = cholesky.GetU();
  TMatrixD UT( U );
  
  UT.Transpose( U );
  
  par = vec_ + UT*tmp;
  
  return par;
}

