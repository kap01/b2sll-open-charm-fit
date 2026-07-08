#ifndef PLOTTING_H
#define PLOTTING_H

#include "TGraph.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"

// Minuit2
#include "Minuit2/FunctionMinimum.h"

// std
#include <vector>

// local 
#include "BESRModel.h" 


namespace Plotting { 

  TGraph* draw( BESRModel& r, const std::vector<double>& par );
  
  TGraph* draw_nonresonant( BESRModel& r, const std::vector<double>& par );

  TGraph* draw_central( BESRModel& r, 
			const ROOT::Minuit2::FunctionMinimum& min );

  TGraph* draw_nonresonant( BESRModel& r, 
			    const ROOT::Minuit2::FunctionMinimum& min );
  
  TGraphErrors* draw_uncertainty( BESRModel& r, 
				  const ROOT::Minuit2::FunctionMinimum& min );
  
  TGraphAsymmErrors* draw_sampled( BESRModel& r, 
				     const ROOT::Minuit2::FunctionMinimum& min,
				     const unsigned int nsamples = 100 );

  TGraph* draw_imhc( BESRModel& r, const std::vector<double>& par );

  TGraph* draw_rehc( BESRModel& r, const std::vector<double>& par );

  double step = 1e-2;
}
  
#endif
