#include "DecayChannel.h" 

int DecayChannel::s_channel = 0; 

DecayChannel::DecayChannel( const double mB, 
			    const double mC, 
			    const unsigned int angmom ) : 
  daughters_( std::make_pair( mB, mC ) ), 
  angmom_   ( angmom )
{
  channel_ = s_channel;  
  s_channel++;
}  

DecayChannel::DecayChannel( const DecayChannel& other ) : 
  daughters_( other.daughters_ ), 
  angmom_   ( other.angmom_    ) 
{
  channel_ = s_channel; 
  s_channel++; 
}

std::pair< double, double > DecayChannel::daughterMasses() const { 
  return std::make_pair( daughters_.first, daughters_.second ) ;
}

unsigned int DecayChannel::angularMomentum() const { 
  return angmom_; 
}

void DecayChannel::add( const int resonance, const double branching ) { 
  resonances_[ resonance ] = branching; 
} 

double DecayChannel::branching( const int resonance ) const { 
  
  auto it = resonances_.find( resonance ) ; 
  
  return ( it != resonances_.end() ? it->second : 0 ) ; 
} 
