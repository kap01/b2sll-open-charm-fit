#ifndef PERTUBATIVEQCD_H
#define PERTUBATIVEQCD_H

#include <complex>

class PerturbativeQCD { 
 public:
  
  PerturbativeQCD() = default; 
  
  ~PerturbativeQCD() = default; 

  double imhc( const double s, const double mc, const double alphas ) const ;
  
  std::complex<double> h0( const double s, const double mc, const double mu ) const;
  
 private:
  double beta( const double s, const double mc ) const ;

  std::complex<double> complex_beta( const double s, const double mc ) const ;
};

#endif
