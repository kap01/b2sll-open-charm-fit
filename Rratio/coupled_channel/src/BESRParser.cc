#include "BESRParser.h" 
#include "DecayChannel.h" 

// Boost
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

// std
#include <iostream>
#include <map> 
#include <algorithm>

// root 
#include "TFile.h" 
#include "TMath.h"
#include "TGraphAsymmErrors.h" 

using boost::property_tree::ptree;

BESRParser::BESRParser(){} 

BESRModel* BESRParser::parse( const std::string filename ) const { 
  
  std::cout 
    << "\n ==> INFO: Parsing model\n" 
    << std::endl;
  
  ptree pt;
  read_json( filename, pt );
  std::map< std::string, DecayChannel* > channels;
  
  // get list of channels
  
  for ( const auto& i : pt.get_child("channels") ){
    channels[ i.first ] = new DecayChannel( i.second.get<double>("M1"),
					    i.second.get<double>("M2"),
					    i.second.get<int>("L") );
  }
  
  // get list of resonances
  
  int ir = 0;

  for ( const auto& i : pt.get_child("resonances") ){
    std::cout << "\t" << i.first << std::endl;
    
    for ( const auto& j: i.second.get_child("channels") ) {
      std::cout << "\t  |-> " << j.first << std::endl;
      
      double bf =  j.second.get<double>("branching-fraction");
      channels[ j.first ]->add( ir, bf );
    }
    
    ir++;
  }

  // Load root file
  std::string datafile = pt.get<std::string>("data");
  
  TFile* tfile = TFile::Open( datafile.c_str() );
  
  TGraphAsymmErrors* data = 
    (TGraphAsymmErrors*) tfile->Get("R");
  
  BESRModel* r = new BESRModel( ir, data );
  
  for ( auto & c: channels ){
    r->open( c.second );
  }
  
  // Fit range
  const double min   = pt.get<double>("min");
  const double max   = pt.get<double>("max");
  
  r->range( min, max );
  
  std::cout << "\n" << std::endl;
  
  return r; 
}

void BESRParser::parameters( const std::string filename, 
			     ROOT::Minuit2::MnUserParameters& par ) const { 
  
  std::cout 
    << "\n ==> INFO: Creating fit parameters\n" 
    << std::endl;
  
  ptree pt;
  read_json( filename, pt );
  
  for ( const auto& i : pt.get_child("resonances") ){
    
    const double m = i.second.get<double>("mass");
    const double w = i.second.get<double>("width");
    
    par.Add( (i.first + "e").c_str(), 5e-7 , 1e-8, 0, 1e-4  );
    par.Add( (i.first + "p").c_str(), 0.0  , 0.1 , -TMath::Pi(), TMath::Pi() );
    par.Add( (i.first + "m").c_str(), m    , 0.01 ); 
    par.Add( (i.first + "w").c_str(), w    , 0.01 ); 
  }
  
  // NR background
  par.Add( "APARAM" , 1.0, 0.1, 0.0, 10.0 );
  
  // Fix the phase of one resonance
  par.Fix( 1 );
  
  return; 
}
