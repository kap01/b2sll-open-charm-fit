#ifndef DISPERSION_H
#define DISPERSION_H

//Local
#include "BESRModel.h" 

//std
#include <complex>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_integration.h>


class Dispersion { 
 public: 
  Dispersion() = default; 
  
  ~Dispersion() = default; 

  double integral( BESRModel& r, 
		   const double t, 
		   const double s, 
		   const double s0, 
		   const std::vector<double>& par ) const ;


  std::complex<double> evaluate( BESRModel& r, 
				 const double s, 
				 const std::vector<double>& par ) const; 
};

#endif
