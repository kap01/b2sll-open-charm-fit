// Local
#include "PerturbativeQCD.h" 

// ROOT
#include "TMath.h" 

// std
#include <cmath>

double PerturbativeQCD::beta( const double s, const double mc ) const { 
  
  if ( s < 4.*mc*mc ) return 0;
  
  return std::sqrt( 1. - 4*mc*mc/s );
}

double PerturbativeQCD::imhc( const double s, 
			      const double mc, 
			      const double alphas ) const { 
  // Schwingers O(alpha_s) result 
  // pg. 6.
  
  if ( s < 4.*mc*mc ) return 0;

  const double v = beta( s, mc );
  
  double result = 0;
  result += (1.0);
  result += (4./3.)*alphas*(0.50*TMath::Pi()/v); 
  result -= (4./3.)*alphas*(0.75 + 0.25*v)*(0.5*TMath::Pi() - 0.75/TMath::Pi());
  result *= (2.*TMath::Pi()*(3. - v*v)*std::abs(v)/9.);
  
  return result;
}

std::complex<double> PerturbativeQCD::complex_beta( const double s, 
						    const double mc ) const { 
  
  return std::sqrt( std::complex<double>(1. - 4*mc*mc/s ) );
}

std::complex<double> PerturbativeQCD::h0( const double s , 
					  const double mc, 
					  const double mu ) const { 
  
  // Eq. A12 
  
  static const std::complex<double> i(0,1);
  
  std::complex<double> result(0,0);
  std::complex<double> option(0,0);
  
  const std::complex<double> v = complex_beta( s, mc );

  result += (4./9.)*((5./3.) - std::norm( v ) - std::log( (mc*mc)/(mu*mu) ));
  
  if ( s < (4*mc*mc) ){
    option = std::atan(1./std::abs(v));
      }
  else { 
    option = 0.5*(std::log((1.+std::abs(v))/(1.-std::abs(v))) - i*TMath::Pi());
  }
  
  result -= (4./9.)*(3. - std::norm(v))*std::abs(v)*option;
  
  return result;
}

