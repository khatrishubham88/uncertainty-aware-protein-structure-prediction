#! /bin/bash
modelname=$1

for CASP_NAME in {7..11}
do 
    echo "====================================================================================="
    echo "============================ CASP $CASP_NAME ================================================="
    python evaluate.py --testdata_path "../proteinnet/data/casp$CASP_NAME/testing/" --model_path "$1" --category 5 --plot --plot_result_dir "test_results_casp$CASP_NAME"
    # python test_plotter.py --testdata_path "../casp$CASP_NAME/testing/" --model_path "$1" --result_dir "test_results_casp$CASP_NAME" --category 5
    zip -r "casp$CASP_NAME\_results.zip" "test_results_casp$CASP_NAME/"
    rm -rf "test_results_casp$CASP_NAME"
    echo "====================================================================================="
    echo
    echo
done
