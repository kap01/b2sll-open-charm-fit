#include "BESRModel.h" 
#include "Variables.h" 

#include <cmath>
#include <limits> 
#include <iostream>



BESRModel::BESRModel( const int nres, TGraphAsymmErrors* data ) : 
  nres_( nres ), 
  data_( data ), 
  def_ ( 1. ),  
  min_ ( std::numeric_limits<double>::min() ),
  max_ ( std::numeric_limits<double>::max() ) {} 


BESRModel::BESRModel( const int nres  ) : 
  nres_( nres ), 
  data_( 0 ), 
  def_ ( 1. ),  
  min_ ( std::numeric_limits<double>::min() ),
  max_ ( std::numeric_limits<double>::max() )  {} 

BESRModel::~BESRModel() {
  channels_.clear() ; 
 } 


double BESRModel::qvalue( const double ma, 
			  const double mb, 
			  const double mc ) const 
{ 
  /*
   *  Rest-frame momentum from a -> b + c decay
   */
  
  if ( ma < 0 || mb < 0 || mc < 0 || ma < ( mb + mc ) ) return 0 ;

  Double_t msq = std::pow(ma,2);
  
  Double_t qsq = (msq - std::pow(mb+mc,2))*(msq - std::pow(mb-mc,2))/(4.*msq);
  
  return std::sqrt( qsq );
}

double BESRModel::barrier( const double qm, 
			   const double qr, 
			   const unsigned int angmom ) const { 
  /*
   * Barrier factor
   */ 
  
  const double qmsq = qm*qm;
  const double qrsq = qr*qr;
  
  if( angmom == 0 ){
    return sqrt((1. + qrsq)/(1. + qmsq));
  }
  
  if( angmom == 2){
    return sqrt((9. + 3.0*qrsq + qrsq*qrsq)/(9. + 3.0*qmsq + qmsq*qmsq));
  }
  
  return 1.0; 
}

double BESRModel::lambda( const double a, const double b, const double c ) const { 
  /*
   * Kallen function
   */
  return ( a*a + b*b + c*c - 2*a*b - 2*a*c - 2*b*c );
}


double BESRModel::gammarunning( const double ma, 
				const double mb, 
				const double mc, 
				const double mean, 
				const double gamma,  
				const unsigned int angmom )  const  { 
  
  if ( ma < (mb + mc) || mean < (mb + mc) ) {
    return 0;
  }
  
  const double radius = Variables::radius;
  
  const double qm = qvalue( ma  , mb, mc ); 
  const double qr = qvalue( mean, mb, mc );

  const double rq  = std::pow( qm/qr, 2*angmom + 1 ); 
  const double rf  = barrier( qr*radius, qm*radius, angmom ); 
  
  return rq*rf*rf*(mean/ma)*gamma; 
}  


double BESRModel::partialwave( const double z, 
			       const unsigned int angmom ) const { 
  
  
  const double zsq = z*z;
  
  if ( 0 == angmom ) {
    return 1.0;
  }
  
  if ( 1 == angmom ){ 
    return 1. + zsq;
  }
  
  if ( 2 == angmom ){
    return 9. + 3.*zsq + zsq*zsq; 
  }
  
  if ( 3 == angmom ){ 
    return 225. + 45.*zsq + 6.*zsq*zsq + zsq*zsq*zsq; 
  }
  
  return 0.0; 
}

double BESRModel::gammapartialwave( const double m , 
				    const double mb, 
				    const double mc, 
				    const double mean, 
				    const double gamma,  
				    const unsigned int angmom ) const  { 
  
  
  const double z = qvalue( m, mb, mc )*Variables::radius;
  
  double result = 0;

  for ( unsigned int l = 0; l <= angmom; ++l ){
    result += std::pow(z,2*l+1)/partialwave(z,l);
  }
  
  return ( 2*gamma*result*mean/(mean + m) );
}

double BESRModel::gammarunning( const double m, 
				const double mean, 
				const double gammaw,
				const int    resonance, 
				const int    channel ) const {
  
  std::pair<double,double> masses = 
    channels_[channel]->daughterMasses();
  
  const unsigned int angmom = 
    channels_[channel]->angularMomentum() ;
  
  const double gamma = 
    gammaw*channels_[channel]->branching( resonance ) ;

  return gammarunning( m, masses.first, masses.second, mean, gamma, angmom );
}


double BESRModel::gammafull( const double m, 
			     const double mean, 
			     const double gammaw, 
			     const double gammae, 
			     const int    resonance ) const {
  
  double gammatot = 0;
  
  gammatot += gammarunning( m, Variables::mElectron, Variables::mElectron, mean, gammae, 0 );
  gammatot += gammarunning( m, Variables::mMuon    , Variables::mMuon    , mean, gammae, 0 );
  gammatot += gammarunning( m, Variables::mTau     , Variables::mTau     , mean, gammae, 0 );
  
  // add hadronic contribution
  for ( unsigned int ic = 0; ic < channels_.size(); ++ic ){ 
    gammatot += gammarunning( m, mean, gammaw, resonance, ic ); 
  }
  
  return gammatot;
}

std::complex<double> BESRModel::bw( const double m, 
				    const double mean, 
				    const double gammae, 
				    const double gammaf,
				    const double gammat,
				    const double phase,
				    const std::complex<double> residue =
				    std::complex<double>(1,0)
				    ) const 
{  
  static const std::complex<double> i( 0 , 1 );
  
  std::complex<double> denom =  mean * mean - m * m - i * mean * gammat ; 
  
  return std::polar( std::sqrt( gammae*gammaf )*mean , phase )*residue/denom ; 
}

std::complex<double> BESRModel::bw( const double m, 
				    const double mean, 
				    const double gammae, 
				    const double gammat ) const {
  
  static const std::complex<double> i( 0 , 1 );
  
  std::complex<double> numer = mean * gammae;
  std::complex<double> denom = mean * mean - m*m - i * mean * gammat ; 
  
  return numer/denom;
}


bool BESRModel::allowed( const double s, const int channel ) const { 
  /* 
   * Check that decay channel is kinematically allowed
   */ 
  std::pair<double,double> masses = 
    channels_[ channel ]->daughterMasses();

  return ( masses.first + masses.second < std::sqrt(s) );
}

double BESRModel::evaluate( const double s, 
			    const std::vector<double>& par ) const 
{
  /*
   * \sigma_{res}    = 12 \pi |T|^2 / s 
   * \sigma_{\mu\mu} = 4  \pi \alpha^2 /3s 
   *
   *
   * R = \sigma_{res.}/\sigma_{\mu\mu} = (9/alpha^2) |T|^2 
   *
   */
  
  double result = 0; 
  double m      = std::sqrt(s);

  for ( unsigned int ic = 0; ic < channels_.size(); ++ic ){ 
    
    std::complex<double> channel(0,0);

    if ( allowed( s, ic ) ){ 
      
      for ( int ir = 0 ; ir < nres_ ; ++ir ) {
	
	double gammae = par[ 4*ir ] ; 
	double phase  = par[ 4*ir + 1 ];
	double mean   = par[ 4*ir + 2 ];
	double gammaw = par[ 4*ir + 3 ];
	
	double gammat = gammafull   ( m, mean, gammaw, gammae, ir );
	double gammar = gammarunning( m, mean, gammaw, ir, ic );  

	channel += bw( m, mean, gammae, gammar, gammat, phase ); 
      }
      
    }

    result += (9./std::pow(Variables::alpha,2))*std::norm( channel );
  }

  result += nonresonant( s, par );
  
  return  result ; 
}


double BESRModel::nonresonant( const double s, 
			       const std::vector<double>& par ) const {
  
  const double Ruds  = 2.16; 
  const double Rudsc = 3.60;  
  
  if ( s < 4.*std::pow(Variables::mDz,2) ) {
    return Ruds; 
  }
  
  const double zc = 4.*std::pow(Variables::mDz,2)/s;
  
  return Ruds + ( 1. - zc )*( Rudsc - Ruds + zc*par[ 4*nres_ ] );
}


void BESRModel::open( DecayChannel* channel ){ 
  channels_.push_back( channel );
}

  
double BESRModel::operator()( const std::vector<double>& par ) const {

  if ( !data_ ) return 0; 
  
  double s, R; 
  double val, low, upp; 

  double chisq = 0 ; 

  for ( int i = 0; i < data_->GetN(); ++i ){ 
    
    data_->GetPoint( i, s , R );
    
    if ( s < min_*min_ || s > max_*max_ ) continue;

    upp = data_->GetErrorYhigh( i );
    low = data_->GetErrorYlow( i ) ; 
    
    val = evaluate( s , par ) ;
    
    if ( val > R ) { 
      chisq += std::pow( (R - val)/upp, 2 );
    }
    else { 
      chisq += std::pow( (R - val)/low, 2 ); 
    }
  }

  return chisq; 
}


void BESRModel::range( const double min, const double max ){ 
  min_ = min;
  max_ = max;
}


std::complex<double> BESRModel::narrowresonance( const double s ) const {
//double BESRModel::narrowresonance( const double s ) const {
  
  std::complex<double> result(0,0);
  
  const double m       = std::sqrt(s);
  const double psi1SBF = 5.96e-2; 
  const double psi2SBF = 7.93e-3;
  
  result += bw( m, Variables::mPsi1S, psi1SBF*Variables::gPsi1S, Variables::gPsi1S );
  result += bw( m, Variables::mPsi2S, psi2SBF*Variables::gPsi2S, Variables::gPsi2S );

  return (9./std::pow(Variables::alpha,2))*result;
  //return (9./std::pow(Variables::alpha,2))*std::imag( result );
}

double BESRModel::lightquark() const { 
   const double Ruds  = 2.16; 
   return Ruds;
}
