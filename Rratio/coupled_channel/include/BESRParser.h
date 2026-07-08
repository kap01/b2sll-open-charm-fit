#ifndef BESRPARSER_H
#define BESRPARSER_H

// local
#include "BESRModel.h" 

// std
#include <string> 

// ROOT 
#include "Minuit2/MnUserParameters.h"

class BESRParser { 
 public:
  BESRParser() ;
  
  BESRModel* parse( const std::string filename ) const ; 
  
  void parameters( const std::string filename, 
		   ROOT::Minuit2::MnUserParameters& par ) const ;
};

#endif 
