https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer
https://cie.co.at/datatable/cie-1964-colour-matching-functions-10-degree-observer

python gen_cmf_header.py <path-to-csv> [float|double]

First arg: full path to the CSV (required).
Second arg: float or double, default double.

Writes the header to stdout — redirect to a file:
python gen_cmf_header.py CIE_xyz_1931_2deg.csv double > CMF_CIE_XYZ_1931_2DEG.hpp
python gen_cmf_header.py CIE_xyz_1964_10deg.csv float  > CMF_CIE_XYZ_1964_10DEG_f.hpp