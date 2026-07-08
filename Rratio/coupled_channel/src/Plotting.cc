#include "Plotting.h"
#include "StatTools.h" 
#include "RandomSampling.h" 
#include "Dispersion.h" 

// ROOT
#include "TMatrixDSym.h"
#include "TVectorD.h" 

#include "Minuit2/MnUserCovariance.h"
#include "Minuit2/MnUserParameters.h"

// std
#include <vector> 
#include <complex>
#include <cmath>
#include <iostream>


TGraph* Plotting::draw( BESRModel& r, const std::vector<double>& par ){ 
  TGraph* result = new TGraph();
  
  double smin = r.smin(); 
  double smax = r.smax();
  double s    = smin;
  
  while ( s <= smax ){ 
    double f = r.evaluate( s, par );
    result->SetPoint( result->GetN(), s, f );
    s += Plotting::step; 
  }
  
  return result; 
}

TGraph* Plotting::draw_nonresonant( BESRModel& r, const std::vector<double>& par ){ 
  TGraph* result = new TGraph();

 
  double smin = r.smin();
  double smax = r.smax();
  double s    = smin;

  while ( s <= smax ){ 
    double f = r.nonresonant( s, par );
    result->SetPoint( result->GetN(), s, f );
    s += Plotting::step; 
  }
  
  return result; 
}

TGraph* Plotting::draw_central( BESRModel& r, 
				const ROOT::Minuit2::FunctionMinimum& min ) { 

  std::vector< double > par = 
    min.UserParameters().Params();

  return Plotting::draw( r, par ); 
}

TGraph* Plotting::draw_nonresonant( BESRModel& r, 
				    const ROOT::Minuit2::FunctionMinimum& min ) { 
  
   std::vector< double > par = 
    min.UserParameters().Params();
   
   return Plotting::draw_nonresonant( r, par );
}


TGraphErrors* Plotting::draw_uncertainty( BESRModel& r, 
					  const ROOT::Minuit2::FunctionMinimum& min )  {
  
  TGraphErrors* result = new TGraphErrors();

  std::vector< double > par = 
    min.UserParameters().Params();

  const unsigned int npar = par.size();
  
  const ROOT::Minuit2::MnUserCovariance& mat = 
    min.UserCovariance();

  TMatrixDSym cor( mat.Nrow() );
  TVectorD    vec( mat.Nrow() );
    
  for ( unsigned int i = 0; i < mat.Nrow(); ++i ){
    for ( unsigned int j = 0; j < mat.Nrow(); ++j ){
      cor(i,j) = mat(i,j);
      cor(i,j) /= std::sqrt(mat(i,i));
      cor(i,j) /= std::sqrt(mat(j,j));
    }
  }
  
  cor.Print();

  std::vector< double > par_error( npar, 0 );
  std::vector< double > par_prime( npar, 0 );
  
  for ( unsigned int i = 0; i < npar; i++ ){ 
    par_error[i] = min.UserParameters().Error( i );
  }
  
  double smin = std::pow(r.min(),2);
  double smax = std::pow(r.max(),2);
  double s    = smin;
  //  double step = 1e-3;
  
  while ( s <= smax ){ 
    
    unsigned int p = 0;

    for ( unsigned int i = 0; i < npar; ++i ){
      
      for ( unsigned int j = 0; j < npar; ++j ){
	par_prime[j] = par[j];
      }
      
      par_prime[i] = par[i] + par_error[i];
      double fupp  = r.evaluate( s, par_prime ); 
	
      par_prime[i] = par[i] - par_error[i];
      double flow  = r.evaluate( s, par_prime ); 

      if ( std::abs( par_error[i] ) > 0. ){ 
      	vec( p++ ) = 0.5*(fupp - flow);
      }
    }
    
    double fnerr  = vec*(cor*vec);
    double fn     = r.evaluate( s, par );
    int ipoint    = result->GetN();
    
    result->SetPoint     ( ipoint, s       , fn );
    result->SetPointError( ipoint, 0.5*step, fnerr );

    s += Plotting::step;
  }
  
  return result;
}


TGraphAsymmErrors* Plotting::draw_sampled( BESRModel& r, 
					   const ROOT::Minuit2::FunctionMinimum& min,
					   const unsigned int nsamples ) { 
 
  const ROOT::Minuit2::MnUserCovariance& mat = 
    min.UserCovariance();
  
  std::vector< double > par = 
    min.UserParameters().Params();
  
  TMatrixDSym cov( mat.Nrow() );
  TVectorD    vec( mat.Nrow() );

  for ( unsigned int i = 0; i < mat.Nrow(); ++i ){
    for ( unsigned int j = 0; j < mat.Nrow(); ++j ){
      cov(i,j) = mat(i,j);
    }
  }
  
  unsigned int p = 0;
  
  for ( const auto & param: min.UserParameters().Parameters() ){ 
    if ( !( param.IsFixed() || param.IsConst() ) ){ 
      vec( p++ ) = par[ param.Number() ];
    }
  }

  RandomSampling rndm( vec, cov );
  
  std::vector< TGraph* > graphs;
  
  for ( unsigned int i = 0; i < nsamples; i++ ){
    graphs.push_back( Plotting::draw( r, par ) );
  
    TVectorD rndm_par = rndm.random();
    
    p = 0;
    
    for ( const auto & param: min.UserParameters().Parameters() ){ 
      if ( !( param.IsFixed() || param.IsConst() ) ){
	par[ param.Number() ] = rndm_par( p++ );
      }
    }
  }
  
  
  TGraphAsymmErrors* result = StatTools::band( graphs, 0.68 );
  
  for ( auto g: graphs ){
    delete g;
  }
  
  return result; 
}



TGraph* Plotting::draw_imhc( BESRModel& r, const std::vector<double>& par ){ 
  TGraph* result = new TGraph();
  
  double smin = r.smin();
  double smax = r.smax();
  double s    = smin;
    
  while ( s <= smax ){ 
    double f = r.evaluate( s, par ) - r.lightquark() + std::imag(r.narrowresonance( s ));
    //double f = r.evaluate( s, par ) - r.lightquark() + r.narrowresonance( s );
    result->SetPoint( result->GetN(), s, f );
    s += Plotting::step; 
  }
  
  return result; 
}

TGraph* Plotting::draw_rehc( BESRModel& r, const std::vector<double>& par ){ 
  TGraph* result = new TGraph();
  
  double smin = r.smin();
  double smax = r.smax();
  double s    = smin;
  
  Dispersion disp;
  
  while ( s <= smax ){ 
    std::complex<double> f = disp.evaluate( r, s, par );
    
    result->SetPoint( result->GetN(), s, std::real(f) );
    s += Plotting::step; 
  }
  
  return result; 
}
