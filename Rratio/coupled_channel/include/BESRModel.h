#ifndef BESRMODEL_H 
#define BESRMODEL_H 

#include "Variables.h" 
#include "DecayChannel.h"

#include <complex> 
#include <vector> 
#include <algorithm>

#include "TGraphAsymmErrors.h" 
#include "Minuit2/FCNBase.h"



class BESRModel : public ROOT::Minuit2::FCNBase { 

 public: 

  BESRModel( const int nres ) ;
  
  BESRModel( const int nres, TGraphAsymmErrors* data ) ;

  ~BESRModel(); 
  
  double evaluate( const double s, const std::vector<double>& par ) const ; 

  double operator()( const std::vector<double>& par ) const  ; 

  virtual double Up() const{ return def_; }
  
  void setErrDef( const double def ) { def_ = def; }
  
  void range( const double min, const double max ) ; 

  void open( DecayChannel* channel ) ; 

  double nonresonant( const double s, const std::vector<double>& par ) const    ; 

  double  min() const { return min_ ; } 
  
  double  max() const { return max_ ; }

  double smin() const { return min_*min_; } 

  double smax() const { return max_*max_; }

  TGraphAsymmErrors* data() { return data_ ; }

  std::complex<double> narrowresonance( const double s ) const ;
  //double narrowresonance( const double s ) const ;
  

  double lightquark() const ;

 private:
  
  std::complex<double> bw(  const double m, 
			    const double mean, 
			    const double gammae, 
			    const double gammaf,
			    const double gammat,
			    const double phase,
			    const std::complex<double> residue) const ;
  
  std::complex<double> bw(  const double m, 
			    const double mean, 
			    const double gammae, 
			    const double gammat ) const ;

  double barrier( const double qm, 
		  const double qr, 
		  const unsigned int angmom ) const; 
  
  double qvalue( const double ma, 
		 const double mb, 
		 const double mc ) const; 

  double lambda( const double a, 
		 const double b, 
		 const double c ) const ;
  
  double partialwave( const double z, 
		      const unsigned int angmom ) const ;

  double gammapartialwave( const double m , 
			   const double mb, 
			   const double mc, 
			   const double mean, 
			   const double gamma,  
			   const unsigned int angmom ) const ;
    
  double gammarunning( const double ma, 
		       const double mb, 
		       const double mc, 
		       const double mean, 
		       const double gamma, 
		       const unsigned int angmom )  const ; 

  double gammafull( const double m, 
		    const double mean, 
		    const double gammaw, 
		    const double gammae, 
		    const int    resonance ) const ; 

  double gammarunning( const double m,
		       const double mean, 
		       const double gammaw,
		       const int    resonance, 
		       const int    channel ) const ;

  
  bool   allowed( const double s, 
		  const int    channel ) const;
  


 private:

  const int nres_ ; 
  
  TGraphAsymmErrors* data_ ;
  
  double def_ ; 

  double min_ ;
  double max_ ; 

  std::vector< DecayChannel* > channels_; 

};


#endif
