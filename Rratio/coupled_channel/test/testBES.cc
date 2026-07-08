#include "../include/BESRModel.h" 
#include "../include/BESRParser.h" 
#include "../include/Variables.h" 
#include "../include/Plotting.h" 


#include "TFile.h" 
#include "TLine.h" 
#include "TGraph.h"
#include "TGraphAsymmErrors.h" 
#include "TGraphErrors.h" 
#include "TAxis.h" 
#include "TMath.h" 
#include "TCanvas.h" 

#include "Minuit2/MnUserParameters.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserCovariance.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnHesse.h"
#include "Minuit2/MnMinos.h"
#include "Minuit2/MnPrint.h"

#include <vector> 
#include <iostream>

void  initialise( ROOT::Minuit2::MnUserParameters& par ) { 
 

  par.Add( "3770e" , 2e-7, 1e-8, 0, 1e-4  );
  par.Add( "3770p" , 0.0 ); // -TMath::Pi(), TMath::Pi() );
  par.Add( "3770m" , 3.771, 0.01 ); //, 3.7, 3.8 );
  par.Add( "3770w" , 0.025, 0.01 ); // , 0.01, 0.04 );

  par.Add( "4040e" , 1e-6, 1e-8, 0, 1e-4  );
  par.Add( "4040p" , 0.0 , 0.1,   -TMath::Pi(), TMath::Pi() );
  par.Add( "4040m" , 4.040, 0.01 ); // , 4.00, 4.10 );
  par.Add( "4040w" , 0.080, 0.01 ); // , 0.01, 0.20 );
  
  par.Add( "4160e" , 1e-6, 1e-8, 0.0, 1e-4  );
  par.Add( "4160p" , 0.0 , 0.1,  -TMath::Pi(), TMath::Pi() );
  par.Add( "4160m" , 4.190, 0.01 ); //, 4.10, 4.30 );
  par.Add( "4160w" , 0.070, 0.01 ); //, 0.01, 0.20 );

  par.Add( "4415e" , 1e-6, 1e-8, 0, 1e-4  );
  par.Add( "4415p" , 0.0 , 0.1,  -TMath::Pi(), TMath::Pi() );
  par.Add( "4415m" , 4.415, 0.01 ); //, 0.01, 4.300, 4.600 );
  par.Add( "4415w" , 0.070, 0.01 ); //, 0.01, 0.01, 0.20 );
  
  par.Add( "APARAM" , 1.0, 0.1, 0.0, 10.0 );
  
  return ;
}


void addNaiveExpectation( const double minval, const double maxval ){ 
  const double uquark = 3.*std::pow(2./3.,2);
  const double dquark = 3.*std::pow(1./3.,2);
  const double squark = 3.*std::pow(1./3.,2);
  const double cquark = 3.*std::pow(2./3.,2);

  const double udvalue   = uquark   + dquark;
  const double udsvalue  = udvalue  + squark;
  const double udscvalue = udsvalue + cquark;
  
  TLine* line_ud   = new TLine( minval*minval, udvalue  , maxval*maxval, udvalue );
  TLine* line_uds  = new TLine( minval*minval, udsvalue , maxval*maxval, udsvalue );
  TLine* line_udsc = new TLine( minval*minval, udscvalue, maxval*maxval, udscvalue ); 

  line_ud  ->Draw(); line_ud  ->SetLineStyle( kDashed );
  line_uds ->Draw(); line_uds ->SetLineStyle( kDashed );
  line_udsc->Draw(); line_udsc->SetLineStyle( kDashed );

  return;
}

void testBES() { 
  BESRParser parser;
  BESRModel* res = parser.parse("test.json");

  const double minval = res->min();
  const double maxval = res->max();
  
  ROOT::Minuit2::MnUserParameters par;
  parser.parameters("test.json",par); //initialise( par );
  
  ROOT::Minuit2::MnMigrad migrad( *res, par );
  
  std::cout << " ==> INFO: Calling MIGRAD \n" << std::endl;
  
  ROOT::Minuit2::FunctionMinimum min = migrad( 100000 ); 
  
  // ROOT::Minuit2::MnHesse hesse; 
  // hesse( *res, min ); 
  
  std::cout << " ==> INFO: Function minimum" << std::endl;
  std::cout << min << std::endl;
  std::cout << std::endl;
    
  std::vector<double> params =  min.UserParameters().Params();

  //std::vector<double> params = par.Params() ; 

  TCanvas* can_fit = new TCanvas("can_fit","can_fit");
  
  TGraph* graph = nullptr; 

  if ( min.IsValid() ){ 
    graph = Plotting::draw_sampled( *res, min, 200 );
  }
  else { 
    graph = Plotting::draw_central( *res, min );
  }
  
  TGraph* graph_nonres = 
    Plotting::draw_nonresonant( *res, min );
  
  graph->Draw("A2");
  graph->SetFillColor( kGreen );
  graph->SetLineColor( kGreen + 2 );
  graph->SetMarkerSize( 0 );
  graph->SetMaximum(5.0);
  graph->SetMinimum(0.0);
  
  graph->GetXaxis()->SetTitle("#it{s} [GeV^{2}]");
  graph->GetYaxis()->SetTitle("R");
  
  //graph->Draw("2+");

  graph_nonres->Draw("L+");
  graph_nonres->SetLineColor( kBlue );
  graph_nonres->SetLineWidth( 1 ) ;
  
  graph->Draw("AL");
  
  res->data()->Draw("P+");
  
  addNaiveExpectation( minval, maxval );
  
  can_fit->Update();
  TCanvas* can_imhc = new TCanvas("can_imhc","can_imhc");
  
  TGraph* gr_imhc = Plotting::draw_imhc( *res, params );
  gr_imhc->GetYaxis()->SetTitle("Im[h(q^{2})]");
  gr_imhc->GetXaxis()->SetTitle("q^{2} (GeV^{2}/#it{c}^{4})");
  gr_imhc->SetMaximum(5.0);
  gr_imhc->Draw("AL");
  can_imhc->Update();

  TCanvas* can_rehc = new TCanvas("can_rehc","can_rehc");
  TGraph* gr_rehc = Plotting::draw_rehc( *res, params );
  gr_rehc->GetYaxis()->SetTitle("Re[h(q^{2})]");
  gr_rehc->GetXaxis()->SetTitle("q^{2} (GeV^{2}/#it{c}^{4})");
  gr_rehc->SetMinimum(-3.);
  gr_rehc->SetMaximum(+3.);
  gr_rehc->Draw("AL");
  

  // now pull out anything that is non DstD...
  TCanvas* can_dstd_only_imhc = new TCanvas("can__dstd_only_imhc",
					    "can_dstd_only_imhc");
  BESRModel* res_dstd_only = parser.parse("test_dstd_only.json");
  TGraph* gr_dstd_only_imhc = Plotting::draw_imhc( *res_dstd_only, params );
  gr_dstd_only_imhc->SetMaximum(5.0);
  gr_dstd_only_imhc->GetYaxis()->SetTitle("Im[h(q^{2})]");
  gr_dstd_only_imhc->GetXaxis()->SetTitle("q^{2} (GeV^{2}/#it{c}^{4})");
  gr_dstd_only_imhc->Draw("AL");

  TCanvas* can_dstd_only_rehc = new TCanvas("can__dstd_only_rehc",
					    "can_dstd_only_rehc");
  TGraph* gr_dstd_only_rehc = Plotting::draw_rehc( *res_dstd_only, params );
  gr_dstd_only_rehc->SetMinimum(-3.);
  gr_dstd_only_rehc->SetMaximum(+3.);
  can_dstd_only_imhc->Update();
  gr_dstd_only_rehc->GetYaxis()->SetTitle("Re[h(q^{2})]");
  gr_dstd_only_rehc->GetXaxis()->SetTitle("q^{2} (GeV^{2}/#it{c}^{4})");
  gr_dstd_only_rehc->Draw("AL");
  
  return;
}
