#ifndef STATTOOLS_H
#define STATTOOLS_H

// std
#include <vector> 

// ROOT
#include "TGraphAsymmErrors.h" 


namespace StatTools {
  
  extern std::pair<double,double> yinterval( std::vector<double>& values, 
					     const double CL = 0.68 ); 

  extern std::pair<double,double> xinterval( const int ipoint, const TGraph* graph );

  
  extern TGraphAsymmErrors* band( std::vector< TGraph* > graphs, 
				  const double CL = 0.68 );
}

#endif
