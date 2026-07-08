
// Local
#include "Dispersion.h" 
#include "PerturbativeQCD.h" 

// ROOT
#include "Math/Functor.h" 
#include "Math/Integrator.h"
#include "TMath.h"

// std
#include <functional>
#include <chrono>
#include <algorithm> 

using namespace std::chrono; 
using namespace std::placeholders;

struct IntegralUtils {
  BESRModel r;
  double s;
  double s0;
  const std::vector<double> bes_par; 
};


double IntegralFunction(double x, void *par)
{
  struct IntegralUtils *params = (struct IntegralUtils *)par;
  double imhc =  ( params->r.evaluate( x, params->bes_par ) - 
		   params->r.lightquark() );
  ////+			 r.narrowresonance( t ) );

  return imhc/(x - params->s0)/(x - params->s);
}

double Dispersion::integral( BESRModel& r, 
			     const double t, 
			     const double s, 
			     const double s0, 
			     const std::vector<double>& par ) const { 

  const double imhc =  ( r.evaluate( t, par ) - 
			 r.lightquark() );
					    ////+			 r.narrowresonance( t ) );

  return imhc/(t - s0)/(t - s);
}

std::complex<double> Dispersion::evaluate( BESRModel& r, 
					   const double s, 
					   const std::vector<double>& par ) const { 
  
  if ( s < r.smin() || s > r.smax() ) {
    return std::complex<double>(0,0);
  }
  
  // Evaluate dispersion integral, Eq. 3

  const double s0   = 3.00;
  const double tmin = std::pow(3.01,2); 
  const double tmax = 50.0; 
  const double eps  = 1e-6;
  /*
  auto integral_func = 
    std::bind( &Dispersion::integral, this, r, _1, s, s0, par );
  
  ROOT::Math::Functor1D  fn( integral_func );
  ROOT::Math::Integrator ig( ROOT::Math::IntegrationOneDim::kADAPTIVE ); //ADAPTIVESINGULAR );
  ig.SetRelTolerance(0.001);
  // kADAPTIVESINGULAR has problems with convergence below J/psi. 
  // kGAUSS is slow.
  // kNONADAPTIVE fails to reach tolerance and performs badly. 
  ig.SetFunction( fn );
  // Evaluate Cauchy principal value by 
  // intrgating up to eps from pole
  double integral_val = 0;
  integral_val += ig.Integral( tmin , s-eps );
  integral_val += ig.Integral( s+eps, tmax  );
  integral_val *= (s - s0)/3.;
  */
  gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000000);
  gsl_function gsl_func;
  struct IntegralUtils int_utils = {r,s,s0,par};
  gsl_func.params = &int_utils;
  gsl_func.function = IntegralFunction;
  double integral_val =0 ;
  double int_val=0,int_err=0;
  auto start = high_resolution_clock::now(); 
  gsl_integration_qags(&gsl_func, tmin, s-eps,1e-3,1e-3,50000,
		       w, &int_val, &int_err);
  integral_val+=int_val;
  gsl_integration_qags(&gsl_func, s+eps, tmax,1e-3,1e-3,50000,
		       w, &int_val, &int_err);
  integral_val += int_val;
  integral_val *= (s - s0)/3.;
  gsl_integration_workspace_free(w);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "Time taken by function: "
	    << duration.count() << " milliseconds" << std::endl; 

  std::complex<double> result(0,0);
  
  PerturbativeQCD pqcd;
  
  result += pqcd.h0( s0, Variables::mcquark, Variables::mbquark ); 
  //result += integral;
  result += integral_val +  r.narrowresonance( s ) ; 
  
  return result; 
}


