To fit the R ratio data, 

```
mkdir build; cd build
cmake ../.
cd ../test 
```

Start ROOT and do the following

```
gSystem->Load("../build/libRModelLib.so");
.x testBES.cc++
```

The fit result will be displaced along with Im(hc) and Re(hc) computed according to arXiv:1406.0566. 

The states used in the fit, their starting parameters and the input data file are currently specified in `test/test.json`. The input data is a TGraphAsymmErrors read from a ROOT file. The R ratio fit can be found in `BESRModel.{cc,h}`. The dispersion realtion calculation for Re(hc) is in `Dispersion.{cc,h}`. Pertubative QCD inputs are given in `PertubativeQCD.{cc,h}`.



