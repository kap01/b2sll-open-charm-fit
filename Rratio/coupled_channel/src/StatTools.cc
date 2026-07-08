#include "StatTools.h"  

// std
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>

std::pair<double,double> StatTools::yinterval( std::vector<double>& values, 
					      const double CL ){
    
    std::sort( values.begin(), values.end() );
    
    unsigned int lowval = (0.5*(1.0-CL))*values.size();
    unsigned int uppval = (1.0 - 0.5*(1.0-CL))*values.size();
    
    return std::make_pair( values[ lowval ], values[ uppval ] ) ;
}

std::pair<double,double> StatTools::xinterval( const int ipoint, const TGraph* graph ){
  
  if ( graph->GetN() < 2 ){
    return std::make_pair( 0, 0 );
  }
  
  const double* x = graph->GetX();
  
  double low = 0;
  double upp = 0;

  if ( ipoint > 0 ){ 
    low = 0.5*( x[ ipoint ] - x[ ipoint - 1 ] );
  }
  if ( ipoint < graph->GetN() - 1 ){
    upp = 0.5*( x[ ipoint + 1 ] - x[ ipoint ] );
  }
  
  return std::make_pair( low, upp );
}


TGraphAsymmErrors* StatTools::band( std::vector< TGraph* > graphs, 
				    const double CL ) {
  
  TGraphAsymmErrors* graph_band = new TGraphAsymmErrors();
  
  if ( graphs.empty() ) return 0;
  
  const unsigned int npoints  = graphs[0]->GetN();
  const unsigned int nsamples = graphs.size();
    
  std::vector<double> values( nsamples, 0 );
  
  double x = 0;
  double y = 0;
  
  for ( unsigned int ipoint = 0; ipoint < npoints; ipoint++ ){
    
    for ( unsigned int igraph = 0; igraph < nsamples; igraph++ ){
      graphs[igraph]->GetPoint( ipoint, x, values[igraph] );
    }
    
    auto xrange = StatTools::xinterval( ipoint, graphs[0] );
    auto yrange = StatTools::yinterval( values, CL );
    
    y = 0.5*( yrange.second + yrange.first );
    
    yrange.second = yrange.second - y;
    yrange.first  = y - yrange.first;
    

    graph_band->SetPoint( ipoint, x, y );
    
    graph_band->SetPointError( ipoint, 
			       xrange.first, xrange.second, 
			       yrange.first, yrange.second );
  }
  
  graph_band->SetFillColor( kGreen );
  
  return graph_band;
}
