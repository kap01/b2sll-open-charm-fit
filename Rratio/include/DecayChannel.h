#ifndef DECAYCHANNEL_H
#define DECAYCHANNEL_H

#include <map> 

class DecayChannel { 
 public: 
  
  DecayChannel( const double mB, const double mC, const unsigned int angmom );
  
  DecayChannel( const DecayChannel& other ); 

  std::pair<double,double> daughterMasses() const ;

  unsigned int angularMomentum() const ;  
  
  void add( const int resonance, const double branching ); 

  double branching( const int resonance ) const ; 

  static int s_channel;  

 private : 

  std::pair<double,double> daughters_;

  unsigned int angmom_;
  
  int channel_ ;

  std::map< int, double > resonances_;
  
};

#endif
