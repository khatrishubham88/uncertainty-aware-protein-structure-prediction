#! /bin/bash
modelname=$1
sample=100
categ=2

for CASP_NAME in {7..11}
do 
    echo "====================================================================================="
    echo "============================ CASP $CASP_NAME ================================================="
    python src/evaluate.py --testdata_path "proteinnet/data/casp"$CASP_NAME"/testing/" --model_path "$1" --category "$categ" --mc --sampling "$sample" --plot --plot_result_dir "test_results_casp"$CASP_NAME
    # python src/test_plotter.py --testdata_path "proteinnet/data/casp$CASP_NAME/testing/" --model_path "$1" --result_dir "test_results_casp$CASP_NAME" --category 5
    zip -r "casp"$CASP_NAME"_results.zip" "test_results_casp"$CASP_NAME"/"
    rm -rf "test_results_casp$CASP_NAME"
    echo "====================================================================================="
    echo
    echo
done
