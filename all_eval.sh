#! /bin/bash
modelname=$1

echo "====================================================================================="
echo "============================ CASP 7 ================================================="
python test_plotter.py --testdata_path ../casp7/testing/ --model_path "$1/chkpnt" --result_dir test_results_casp7 --category 5
zip -r casp7_results.zip test_results_casp7/
rm -rf test_results_casp7
echo "====================================================================================="
echo
echo

echo "====================================================================================="
echo "============================ CASP 8 ================================================="
python test_plotter.py --testdata_path ../casp8/testing/ --model_path "$1/chkpnt" --result_dir test_results_casp8 --category 5
zip -r casp8_results.zip test_results_casp8/
rm -rf test_results_casp8
echo "====================================================================================="
echo
echo

echo "====================================================================================="
echo "============================ CASP 9 ================================================="
python test_plotter.py --testdata_path ../casp9/testing/ --model_path "$1/chkpnt" --result_dir test_results_casp9 --category 5
zip -r casp9_results.zip test_results_casp9/
rm -rf test_results_casp9
echo "====================================================================================="
echo
echo

echo "====================================================================================="
echo "============================ CASP 10 ================================================"
python test_plotter.py --testdata_path ../casp10/testing/ --model_path "$1/chkpnt" --result_dir test_results_casp10 --category 5
zip -r casp10_results.zip test_results_casp10/
rm -rf test_results_casp10
echo "====================================================================================="
echo
echo

echo "====================================================================================="
echo "============================ CASP 11 ================================================="
python test_plotter.py --testdata_path ../casp11/testing/ --model_path "$1/chkpnt" --result_dir test_results_casp11 --category 5
zip -r casp11_results.zip test_results_casp11/
rm -rf test_results_casp11
echo "====================================================================================="
echo "Done!!!!"
echo
