https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer
https://cie.co.at/datatable/cie-1964-colour-matching-functions-10-degree-observer


python gen_cct_lut_header_cpp14.py CIE_xyz_1931_2deg.csv 1931 --cct-min 1000 --cct-max 26010 --cct-step 1
python gen_cct_lut_header_cpp14.py CIE_xyz_1964_10deg.csv 1964 --cct-min 1000 --cct-max 26010 --cct-step 1

python gen_cct_lut_header_cpp20.py CIE_xyz_1931_2deg.csv 1931 --cct-min 1000 --cct-max 26010 --cct-step 1
python gen_cct_lut_header_cpp20.py CIE_xyz_1964_10deg.csv 1964 --cct-min 1000 --cct-max 26010 --cct-step 1